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


# sm_60 GPUs: incompatible with torch 2.10+, need torch 2.3.x+cu118
_SM60_GPU_NAMES = ("P100", "K80", "M60", "M40", "K40", "P4", "P40")
_TORCH_SM60 = "torch==2.3.1+cu118"
_TORCH_SM60_INDEX = "https://download.pytorch.org/whl/cu118"


def _gpu_name() -> str:
    """Return the first GPU name from nvidia-smi, or empty string."""
    for args in (
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        ["nvidia-smi", "-L"],
    ):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0].strip()
    return ""


def check_gpu():
    """Install a compatible torch version if Kaggle assigned a sm_60 GPU (P100, K80…).

    torch 2.10 requires sm_70+. Instead of bailing, we install torch 2.3.1+cu118
    which supports sm_60. The subprocess (kaggle_run.py) starts fresh and picks up
    the newly installed version. CUDA 12.x drivers are backwards-compatible with
    cu118 binaries, so this works on Kaggle regardless of driver version.
    """
    name = _gpu_name()
    if not name:
        print("[kernel] Could not query GPU name — proceeding with default torch.")
        return

    if any(old in name for old in _SM60_GPU_NAMES):
        print(
            f"[kernel] sm_60 GPU detected ({name}). "
            f"Installing {_TORCH_SM60} for compatibility…"
        )
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q",
                _TORCH_SM60,
                "--index-url", _TORCH_SM60_INDEX,
            ],
            check=True,
        )
        print(f"[kernel] {_TORCH_SM60} installed — training will proceed on {name}.")
    else:
        print(f"[kernel] GPU OK: {name}")


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
    check_gpu()
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
