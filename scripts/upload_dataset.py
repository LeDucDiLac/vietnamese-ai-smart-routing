#!/usr/bin/env python3
"""Push data/processed/v2 as a new version of duckgotsick/ai-smart-routing-dataset on Kaggle.

Run from repo root on this machine (before training on the H200):
    python scripts/upload_dataset.py
    python scripts/upload_dataset.py --version-dir data/processed/v2 -m "v2: updated labels"

Then on the H200, download the latest version with:
    kaggle datasets download duckgotsick/ai-smart-routing-dataset --unzip -p ~/data/ai-smart-routing/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET_ID = "duckgotsick/ai-smart-routing-dataset"
DEFAULT_VERSION_DIR = REPO / "data" / "processed" / "v2"


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Push a new version of the dataset to Kaggle")
    ap.add_argument(
        "--version-dir",
        default=str(DEFAULT_VERSION_DIR),
        help=f"Directory containing train/val/test.jsonl (default: {DEFAULT_VERSION_DIR})",
    )
    ap.add_argument(
        "-m", "--message",
        default="v2: updated dataset",
        help="Version message shown on Kaggle",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the kaggle command but do not run it",
    )
    args = ap.parse_args()

    vdir = Path(args.version_dir).expanduser().resolve()

    if not vdir.is_dir():
        print(f"ERROR: version dir not found: {vdir}", file=sys.stderr)
        sys.exit(1)

    # Validate expected files
    expected = ("train.jsonl", "val.jsonl", "test.jsonl")
    missing = [f for f in expected if not (vdir / f).exists()]
    if missing:
        print(f"ERROR: missing files in {vdir}: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset directory: {vdir}")
    for fname in expected:
        n = _count_rows(vdir / fname)
        print(f"  {fname}: {n:,} rows")

    # Ensure dataset-metadata.json exists
    meta_path = vdir / "dataset-metadata.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {"title": "AI Smart Routing Dataset", "id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}]},
                indent=2,
            )
        )
        print(f"  Created {meta_path.name}")

    cmd = [
        "kaggle", "datasets", "version",
        "-p", str(vdir),
        "-m", args.message,
        "--dir-mode", "zip",
    ]
    print(f"\n$ {' '.join(cmd)}")

    if args.dry_run:
        print("(dry-run — not executed)")
        return

    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print("\nERROR: kaggle datasets version failed.", file=sys.stderr)
        sys.exit(r.returncode)

    print(f"\nDone. View at: https://www.kaggle.com/datasets/{DATASET_ID}")
    print("\nOn the H200 server, download with:")
    print(f"  kaggle datasets download {DATASET_ID} --unzip -p ~/data/ai-smart-routing/")
    print("\nThen train:")
    print("  python scripts/train_all_teachers.py --data-root ~/data/ai-smart-routing/")


if __name__ == "__main__":
    main()
