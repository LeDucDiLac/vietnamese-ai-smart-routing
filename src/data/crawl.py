"""Crawl / download Vietnamese corpora into (optionally weak-labeled) prompts.

Pulls candidate HuggingFace datasets (from ``configs/data_sources.yaml``, the
probe-verified registry), extracts the prompt-like field from each, **cleans**
the text (strip markup/entities, length bounds, drop noise), language-filters
(keep Vietnamese + mixed vi/en, drop pure-en/zh), dedupes, and writes one JSONL
of prompts to ``data/raw/``.

Three labeling tiers (plan §3.1 / §3.2):

- **none** (default): raw ``{"text", "source"}`` rows — input to the GPU/paid
  teacher labeling stage.
- **provenance** (``--label provenance``): stamp each row with the source's
  ``task_hint`` as a *weak* ``task_type``. Costs $0, no GPU, no API. Reliable for
  single-purpose sources (vietnews -> summarization, ViQuAD -> closed_qa,
  feedback -> classification); approximate for broad instruction sets. This is a
  **balanced labeling pool for inspection / teacher seeding, NOT a training set**
  — it carries no complexity dimensions, which only a teacher can produce.

Quotas (so the output is balanced, not skewed to whatever's abundant):

- ``--max-per-source``  cap of *kept* rows per dataset (post-filter; filtered-out
  rows don't count toward it).
- ``--per-label``       cap of *kept* rows per task_hint label across all sources
  (provenance mode) — stop collecting a label once it's full.

Needs the ``data`` extra (``datasets`` + ``huggingface-hub``); language filtering
uses a stdlib heuristic by default (no model download) or an optional fastText
detector if installed.

    python -m data.crawl --out data/raw/crawl.jsonl \
        --label provenance --max-per-source 5000 --per-label 2000
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from config import CONFIGS_DIR, DATA_DIR, load_label_schema, task_type_id

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """One HF dataset to mine, and how to pull a prompt string out of a row."""

    hf_id: str
    text_fields: tuple[str, ...]  # try each in order; first non-empty wins
    split: str = "train"
    config: str | None = None
    task_hint: str | None = None  # provenance weak label (a task_type id)
    trust_remote_code: bool = False  # ignored — datasets library removed this; kept for YAML compat
    max_rows: int | None = None      # per-source override for --max-per-source


# Fallback if configs/data_sources.yaml is missing (kept minimal; the YAML is
# the source of truth and is probe-verified).
FALLBACK_SOURCES: list[Source] = [
    Source("bkai-foundation-models/vi-alpaca", ("instruction", "input"),
           task_hint="text_generation"),
    Source("Yuhthe/vietnews", ("article",), task_hint="summarization"),
]


def load_sources(path: str | Path | None = None) -> list[Source]:
    """Load the candidate sources from ``configs/data_sources.yaml``."""
    p = Path(path) if path else CONFIGS_DIR / "data_sources.yaml"
    if not p.exists():
        print(f"  data_sources.yaml not found at {p}; using fallback sources")
        return FALLBACK_SOURCES
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    sources: list[Source] = []
    for s in data.get("sources", []):
        sources.append(
            Source(
                hf_id=s["id"],
                text_fields=tuple(s.get("text_fields", ["text"])),
                split=s.get("split", "train"),
                config=s.get("config"),
                task_hint=s.get("task_hint"),
                trust_remote_code=bool(s.get("trust_remote_code", False)),
                max_rows=s.get("max_rows"),
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
# wiki/markup leftovers seen in vietgpt/wikipedia_vi samples
_MARKUP = re.compile(
    r"(<[^>]+>"                       # html / templatestyles tags
    r"|__[A-Z]+__"                    # __NOEDITSECTION__ etc.
    r"|\{\{[^}]*\}\}"                 # {{templates}}
    r"|\[\[|\]\])"                    # [[wikilinks]]
)
_URL = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Normalize a raw field into clean prompt text.

    Strips HTML/wiki markup + entities, collapses whitespace, removes URLs. Pure
    stdlib (re + html) so it runs anywhere. Returns ``""`` for unsalvageable input
    (caller drops empty results).
    """
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)          # &lt; -> < , &amp; -> & , &#39; -> '
    t = _MARKUP.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def passes_quality(
    text: str, *, min_chars: int, max_chars: int, max_digit_ratio: float = 0.4
) -> bool:
    """Reject too-short/too-long fragments and number/symbol-dominated noise."""
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    if n:
        digits = sum(c.isdigit() for c in text)
        if digits / n > max_digit_ratio:
            return False
    # need at least a few word-like tokens
    if len(text.split()) < 3:
        return False
    return True


# ---------------------------------------------------------------------------
# Language filter
# ---------------------------------------------------------------------------

# Characters that appear in Vietnamese but not in plain ASCII English.
_VN_DIACRITICS = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
_CJK = re.compile(r"[一-鿿぀-ヿ]")


def looks_vietnamese(text: str, *, min_diacritic_ratio: float = 0.003) -> bool:
    """Heuristic: keep Vietnamese + mixed vi/en, drop pure-en/zh/ja.

    Vietnamese text reliably carries diacritics; a tiny ratio of diacritic chars
    to total chars is enough to distinguish it from English. CJK text is dropped
    outright.
    """
    if not text:
        return False
    if _CJK.search(text):
        return False
    n = len(text)
    diac = len(_VN_DIACRITICS.findall(text))
    return (diac / n) >= min_diacritic_ratio if n else False


def _fasttext_is_vi(detector: Any, text: str) -> bool:  # pragma: no cover - optional
    try:
        res = detector.detect(text.replace("\n", " ")[:1000])
        return res.get("lang") == "vi"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------


def _extract_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for f in fields:
        val = row.get(f)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def crawl_source(
    src: Source,
    *,
    max_rows: int,
    min_chars: int,
    max_chars: int,
    use_fasttext: bool,
    label_mode: str,
    detector: Any = None,
    streaming: bool = True,
) -> Iterable[dict[str, Any]]:
    """Yield cleaned (optionally weak-labeled) prompt rows from one source.

    Tolerates a load failure (logs ``[skip]`` and returns). ``max_rows`` counts
    only *kept* rows — cleaned, quality-passed, language-passed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("crawl needs the `data` extra: pip install -e '.[data]'") from exc

    try:
        ds = load_dataset(src.hf_id, src.config, split=src.split, streaming=streaming)
    except Exception as exc:
        print(f"  [skip] {src.hf_id}: {exc}")
        return

    kept = 0
    seen_source = 0
    for row in ds:
        if kept >= max_rows:
            break
        seen_source += 1
        # hard stop so a giant source (e.g. 1.3M-row wiki) can't stream forever
        # looking for kept rows if the filter rejects almost everything.
        if seen_source > max(max_rows * 50, 50_000):
            break
        raw = _extract_text(row, src.text_fields)
        if not raw:
            continue
        text = clean_text(raw)
        if not passes_quality(text, min_chars=min_chars, max_chars=max_chars):
            continue
        keep = (
            _fasttext_is_vi(detector, text) if (use_fasttext and detector)
            else looks_vietnamese(text)
        )
        if not keep:
            continue
        out: dict[str, Any] = {"text": text[:max_chars], "source": src.hf_id}
        if label_mode == "provenance" and src.task_hint:
            out["task_type"] = task_type_id(src.task_hint)
            out["label_source"] = "provenance"
        yield out
        kept += 1
    print(f"  [ok]   {src.hf_id}: {kept} kept (scanned ~{seen_source})")


def _safe_id(hf_id: str, config: str | None = None) -> str:
    """HF dataset ID → safe filename stem (slashes → double-underscore)."""
    name = hf_id.replace("/", "__")
    if config:
        name = f"{name}__{config}"
    return name


def crawl(
    out_path: str | Path,
    *,
    sources: list[Source] | None = None,
    max_per_source: int = 5000,
    per_label: int | None = None,
    min_chars: int = 12,
    max_chars: int = 4000,
    label_mode: str = "none",
    use_fasttext: bool = False,
    source_dir: str | Path | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Crawl all sources into one JSONL; dedupe; enforce per-source & per-label caps.

    When ``source_dir`` is given, each source is also written to its own
    ``{source_dir}/{safe_id}.jsonl`` file alongside the merged output.  Set
    ``skip_existing=True`` to skip re-downloading sources whose per-source file
    already exists (incremental re-crawl when adding new sources).

    Returns a report dict (counts + per-label coverage + uncovered task types).
    """
    sources = sources if sources is not None else load_sources()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    src_dir = Path(source_dir) if source_dir else None
    if src_dir:
        src_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    if use_fasttext:  # pragma: no cover - optional path
        try:
            from ftlangdetect import detect as _detect  # type: ignore

            class _D:
                detect = staticmethod(lambda t: _detect(t))

            detector = _D()
        except ImportError:
            print("  fasttext-langdetect not installed; using heuristic filter")

    seen: set[str] = set()
    per_label_count: dict[str, int] = {}
    written = 0
    label_dist: dict[str, int] = {}
    source_dist: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as fh:
        for src in sources:
            src_path = src_dir / f"{_safe_id(src.hf_id, src.config)}.jsonl" if src_dir else None
            use_cache = skip_existing and src_path is not None and src_path.exists()

            src_max = src.max_rows if src.max_rows is not None else max_per_source
            if use_cache:
                print(f"  [cached] {src.hf_id}: reading from {src_path.name}")  # type: ignore[union-attr]
                row_iter: Iterable[dict[str, Any]] = (
                    json.loads(line)
                    for line in src_path.open(encoding="utf-8")  # type: ignore[union-attr]
                    if line.strip()
                )
            else:
                row_iter = crawl_source(
                    src,
                    max_rows=src_max,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    use_fasttext=use_fasttext,
                    label_mode=label_mode,
                    detector=detector,
                )

            src_fh = src_path.open("w", encoding="utf-8") if (src_path and not use_cache) else None
            try:
                for row in row_iter:
                    # near-dup key: normalized prefix
                    key = " ".join(row["text"].lower().split())[:200]
                    if key in seen:
                        continue

                    # per-label quota (provenance mode)
                    label = row.get("task_type")
                    if per_label is not None and label is not None:
                        if per_label_count.get(label, 0) >= per_label:
                            continue
                        per_label_count[label] = per_label_count.get(label, 0) + 1

                    seen.add(key)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if src_fh:
                        src_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if label is not None:
                        label_dist[label] = label_dist.get(label, 0) + 1
                    source_dist[row["source"]] = source_dist.get(row["source"], 0) + 1
            finally:
                if src_fh:
                    src_fh.close()

    report: dict[str, Any] = {
        "written": written,
        "label_mode": label_mode,
        "per_source": source_dist,
    }
    if label_mode == "provenance":
        schema = load_label_schema()
        all_tasks = set(schema.task_type_ids)
        covered = set(label_dist.keys())
        report["label_distribution"] = label_dist
        report["covered_task_types"] = sorted(covered)
        report["uncovered_task_types"] = sorted(all_tasks - covered)

    (out.parent / "crawl_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {written} prompts to {out}")
    if label_mode == "provenance":
        print(f"  label distribution: {report['label_distribution']}")
        if report["uncovered_task_types"]:
            print(
                "  NOT covered by any crawl source (need synthetic/teacher): "
                f"{report['uncovered_task_types']}"
            )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Crawl Vietnamese corpora into prompts")
    ap.add_argument("--out", default=str(DATA_DIR / "raw" / "crawl.jsonl"))
    ap.add_argument("--sources", default=None, help="path to data_sources.yaml")
    ap.add_argument("--max-per-source", type=int, default=5000,
                    help="cap of KEPT rows per source (post-filter)")
    ap.add_argument("--per-label", type=int, default=None,
                    help="cap of KEPT rows per task_hint label (provenance mode)")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--label", choices=["none", "provenance"], default="none",
                    help="weak-label rows from source task_hint")
    ap.add_argument("--use-fasttext", action="store_true", help="use fastText langdetect")
    ap.add_argument("--source-dir", default=None,
                    help="write per-source JSONL files to this directory alongside merged output")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip re-downloading sources whose per-source file already exists")
    args = ap.parse_args()

    crawl(
        args.out,
        sources=load_sources(args.sources),
        max_per_source=args.max_per_source,
        per_label=args.per_label,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        label_mode=args.label,
        use_fasttext=args.use_fasttext,
        source_dir=args.source_dir,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
