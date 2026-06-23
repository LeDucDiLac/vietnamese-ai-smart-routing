"""Kaggle kernel script — runs the ai-smart-routing training pipeline.

Triggered via `kaggle kernels push`. Clones the repo from GitHub (internet
must be enabled on the kernel), reads run parameters from run_config.json,
then drives kaggle_run.py. All output is teed to /kaggle/working/run.log so
the automation script can retrieve it via `kaggle kernels output`.
"""
import json
import os
import pathlib
import subprocess
import sys

WORKING = pathlib.Path("/kaggle/working")
REPO = WORKING / "ai-smart-routing"
LOG = WORKING / "run.log"
CONFIG = REPO / "kaggle" / "run_config.json"
GITHUB_URL = "https://github.com/LeDucDiLac/vietnamese-ai-smart-routing.git"


def git_clone():
    if REPO.exists():
        print(f"[kernel] repo already present at {REPO}, pulling latest")
        subprocess.run(["git", "-C", str(REPO), "pull"], check=True)
    else:
        print(f"[kernel] cloning {GITHUB_URL}")
        subprocess.run(["git", "clone", GITHUB_URL, str(REPO)], check=True)


def build_cmd(cfg: dict) -> list[str]:
    cmd = [
        sys.executable, "kaggle_run.py",
        "--install",
        "--steps", cfg["steps"],
        "--data-root", cfg["data_root"],
        "--epochs", str(cfg.get("epochs", 3)),
    ]
    if cfg.get("batch_size"):
        cmd += ["--batch-size", str(cfg["batch_size"])]
    if cfg.get("max_steps"):
        cmd += ["--max-steps", str(cfg["max_steps"])]
    cmd += cfg.get("extra_args", [])
    return cmd


def main():
    git_clone()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cmd = build_cmd(cfg)
    print(f"[kernel] command: {' '.join(cmd)}")
    print(f"[kernel] logging to {LOG}")

    with LOG.open("w", buffering=1, encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()

    print(f"[kernel] exit code: {proc.returncode}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
