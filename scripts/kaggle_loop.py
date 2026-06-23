#!/usr/bin/env python3
"""Autonomous training loop: smoke test -> commit/push -> trigger Kaggle -> poll -> fetch logs.

Usage (from repo root):
    python scripts/kaggle_loop.py --kernel duckgotsick/ai-smart-routing-train

Flags:
    --kernel        Kaggle kernel slug, e.g. duckgotsick/ai-smart-routing-train
    --skip-smoke    Skip local smoke test (faster, riskier)
    --skip-push     Skip git commit+push (useful if already pushed)
    --smoke-steps   Max training steps for the local smoke test (default: 20)
    --poll-interval Seconds between Kaggle status polls (default: 60)
    --timeout       Max seconds to wait for Kaggle run (default: 7200 = 2h)
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
KAGGLE_DIR = REPO / "kaggle"
SMOKE_LOG = REPO / "runs" / "smoke_test.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def torch_available() -> bool:
    r = subprocess.run(
        [sys.executable, "-c", "import torch"],
        cwd=REPO, capture_output=True,
    )
    return r.returncode == 0


def local_smoke_test(max_steps: int) -> bool:
    section(f"Local smoke test (--max-steps {max_steps})")
    if not torch_available():
        print("  torch not installed locally — running syntax check instead.")
        return syntax_check()
    SMOKE_LOG.parent.mkdir(parents=True, exist_ok=True)
    r = sh(
        [
            sys.executable, "kaggle_run.py",
            "--steps", "train",
            "--data-root", "data/processed",
            "--epochs", "1",
            "--max-steps", str(max_steps),
        ],
        cwd=REPO,
    )
    if r.returncode == 0:
        print("\n✓ Smoke test passed.")
        return True
    print("\n✗ Smoke test FAILED — fix the error above before pushing to Kaggle.")
    return False


def syntax_check() -> bool:
    """Compile all Python source files to catch syntax errors without torch."""
    import py_compile
    errors = []
    for path in sorted((REPO / "src").rglob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(str(e))
    if errors:
        print("\n✗ Syntax errors found:")
        for e in errors:
            print(f"  {e}")
        return False
    print(f"✓ Syntax OK ({len(list((REPO / 'src').rglob('*.py')))} files checked).")
    return True


def commit_and_push(message: str = "auto: training iteration") -> None:
    section("Commit + push")
    sh(["git", "-C", str(REPO), "add", "-A"])
    r = sh(["git", "-C", str(REPO), "commit", "-m", message])
    if r.returncode == 0:
        sh(["git", "-C", str(REPO), "push", "origin", "main"], check=True)
    else:
        print("Nothing to commit — repo is clean, pushing anyway.")
        sh(["git", "-C", str(REPO), "push", "origin", "main"])


def trigger_kaggle_run(kernel: str) -> None:
    section(f"Push kernel {kernel} to Kaggle")
    sh(["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR)], check=True)
    print(f"\nKernel queued. Track at: https://www.kaggle.com/code/{kernel}")


def poll_until_done(kernel: str, interval: int, timeout: int) -> str:
    section(f"Polling {kernel} (every {interval}s, timeout {timeout//60}m)")
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = sh(
            ["kaggle", "kernels", "status", kernel],
            capture_output=True, text=True,
        )
        line = (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "?"
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] {line}")

        lower = line.lower()
        if "complete" in lower:
            return "complete"
        if "error" in lower or "cancel" in lower or "failed" in lower:
            return "error"

        time.sleep(interval)

    return "timeout"


def fetch_logs(kernel: str) -> None:
    section(f"Fetching output from {kernel}")
    out_dir = REPO / "runs" / "kaggle_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "output", kernel, "-p", str(out_dir)])
    log_path = out_dir / "run.log"
    if log_path.exists():
        print(f"\n{'='*60}\n  run.log\n{'='*60}")
        print(log_path.read_text(encoding="utf-8"))
    else:
        print(f"No run.log in {out_dir}. Files present:")
        for f in sorted(out_dir.iterdir()):
            print(f"  {f.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Autonomous Kaggle training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--kernel", required=True,
        help="Kaggle kernel slug, e.g. duckgotsick/ai-smart-routing-train",
    )
    ap.add_argument("--skip-smoke", action="store_true", help="Skip local smoke test")
    ap.add_argument("--skip-push", action="store_true", help="Skip git commit+push")
    ap.add_argument("--smoke-steps", type=int, default=20, help="Steps for local smoke test")
    ap.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls")
    ap.add_argument("--timeout", type=int, default=7200, help="Max wait seconds for Kaggle run")
    args = ap.parse_args()

    # 1. local smoke test
    if not args.skip_smoke:
        if not local_smoke_test(args.smoke_steps):
            sys.exit(1)

    # 2. commit + push
    if not args.skip_push:
        commit_and_push()

    # 3. trigger kaggle run
    trigger_kaggle_run(args.kernel)

    # 4. poll
    status = poll_until_done(args.kernel, args.poll_interval, args.timeout)

    # 5. fetch logs
    fetch_logs(args.kernel)

    section(f"Done — status: {status.upper()}")
    sys.exit(0 if status == "complete" else 1)


if __name__ == "__main__":
    main()
