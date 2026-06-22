"""Smoke-test candidate corpora against the HF datasets-server (plan §3.1).

Validates every source in ``configs/data_sources.yaml`` WITHOUT downloading the
corpora: it queries the public HuggingFace datasets-server REST API for validity,
splits, and a few sample rows. Writes a manifest to ``data/probe/`` so we know
which IDs actually resolve (and which fields carry prompt text) before a real
crawl spends time/GPU. Gated/404/renamed sources are flagged, not hidden.

Stdlib only (urllib + the project's pyyaml) — no ``datasets`` dependency, runs
anywhere including a thin CI box.

    python -m data.probe --out data/probe
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from config import CONFIGS_DIR, DATA_DIR

DSS = "https://datasets-server.huggingface.co"
UA = {"User-Agent": "ai-smart-routing-probe/0.1"}


def _get(url: str, timeout: float = 25.0) -> tuple[int, dict[str, Any] | None]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = None
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            pass
        return e.code, body
    except Exception as e:  # network / timeout
        return -1, {"error": str(e)}


def _load_sources(path: str | Path | None) -> list[dict[str, Any]]:
    p = Path(path) if path else CONFIGS_DIR / "data_sources.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data["sources"]


def probe_one(src: dict[str, Any], n_samples: int = 3) -> dict[str, Any]:
    """Probe a single source: is it valid, what splits, do the text fields exist."""
    ds_id = src["id"]
    config = src.get("config")
    split = src.get("split", "train")
    want_fields = src.get("text_fields", [])

    result: dict[str, Any] = {
        "id": ds_id,
        "config": config,
        "task_hint": src.get("task_hint"),
        "requested_fields": want_fields,
        "status": "unknown",
    }

    # 1. is-valid
    q = urllib.parse.urlencode({"dataset": ds_id})
    code, valid = _get(f"{DSS}/is-valid?{q}")
    if code == 401 or (valid and valid.get("error", "").lower().find("gated") >= 0):
        result["status"] = "gated"
        result["detail"] = valid
        return result
    if code != 200 or valid is None:
        result["status"] = "unreachable" if code == -1 else f"http_{code}"
        result["detail"] = valid
        return result

    # 2. splits — confirm the requested config/split exists
    code, splits = _get(f"{DSS}/splits?{q}")
    available = []
    if code == 200 and splits:
        for s in splits.get("splits", []):
            available.append({"config": s.get("config"), "split": s.get("split")})
    result["available_splits"] = available[:12]

    use_config = config
    if available and config is None:
        use_config = available[0]["config"]
    if available:
        # pick a split that exists for the chosen config
        cfgs = [a for a in available if a["config"] == use_config]
        if cfgs and not any(a["split"] == split for a in cfgs):
            split = cfgs[0]["split"]

    # 3. sample rows via /rows
    rq = urllib.parse.urlencode(
        {
            "dataset": ds_id,
            "config": use_config or "default",
            "split": split,
            "offset": 0,
            "length": n_samples,
        }
    )
    code, rows = _get(f"{DSS}/rows?{rq}")
    if code != 200 or not rows:
        result["status"] = "valid_no_rows"
        result["detail"] = rows
        return result

    row_list = rows.get("rows", [])
    columns = []
    if row_list:
        columns = list(row_list[0].get("row", {}).keys())
    result["columns"] = columns

    # which requested fields actually exist + carry text
    found = []
    samples = []
    for f in want_fields:
        if f in columns:
            found.append(f)
    # capture a short sample of the first found field
    if found and row_list:
        for r in row_list[:n_samples]:
            val = r.get("row", {}).get(found[0])
            if isinstance(val, str):
                samples.append(val[:160])
    result["resolved_config"] = use_config
    result["resolved_split"] = split
    result["fields_found"] = found
    result["fields_missing"] = [f for f in want_fields if f not in columns]
    result["samples"] = samples
    result["status"] = "ok" if found else "valid_fields_missing"
    return result


def probe_all(
    out_dir: str | Path,
    *,
    sources_path: str | Path | None = None,
    n_samples: int = 3,
) -> dict[str, Any]:
    sources = _load_sources(sources_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for src in sources:
        print(f"  probing {src['id']} ...", flush=True)
        results.append(probe_one(src, n_samples=n_samples))

    summary = {
        "ok": [r["id"] for r in results if r["status"] == "ok"],
        "fields_missing": [
            r["id"] for r in results if r["status"] == "valid_fields_missing"
        ],
        "gated": [r["id"] for r in results if r["status"] == "gated"],
        "broken": [
            r["id"]
            for r in results
            if r["status"] not in {"ok", "valid_fields_missing", "gated"}
        ],
    }
    manifest = {"summary": summary, "results": results}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # also write a ready-to-use crawl source list of only the OK ones
    crawl_ready = [
        {
            "id": r["id"],
            "config": r.get("resolved_config"),
            "split": r.get("resolved_split", "train"),
            "text_fields": r["fields_found"],
            "task_hint": r.get("task_hint"),
        }
        for r in results
        if r["status"] == "ok"
    ]
    (out / "crawl_ready.json").write_text(
        json.dumps(crawl_ready, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test candidate VN corpora")
    ap.add_argument("--out", default=str(DATA_DIR / "probe"))
    ap.add_argument("--sources", default=None, help="path to data_sources.yaml")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    manifest = probe_all(args.out, sources_path=args.sources, n_samples=args.samples)
    s = manifest["summary"]
    print("\n=== probe summary ===")
    print(f"  ok              ({len(s['ok'])}): {s['ok']}")
    print(f"  fields_missing  ({len(s['fields_missing'])}): {s['fields_missing']}")
    print(f"  gated           ({len(s['gated'])}): {s['gated']}")
    print(f"  broken          ({len(s['broken'])}): {s['broken']}")
    print(f"  manifest -> {Path(args.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
