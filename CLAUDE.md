# AI Smart Routing — Claude Code Guide

## Autonomous Kaggle Training Loop

All GPU training runs on Kaggle. To fix code and trigger a new run without manual intervention:

```bash
python scripts/kaggle_loop.py --kernel duckgotsick/ai-smart-routing-train
```

This will:
1. Syntax-check all Python source files (or full smoke test if torch is installed locally)
2. Commit and push changes to GitHub
3. Trigger the Kaggle kernel `duckgotsick/ai-smart-routing-train`
4. Poll every 60s until the run completes or errors
5. Download and print `run.log` so errors can be read and fixed immediately

### Run parameters

Edit [kaggle/run_config.json](kaggle/run_config.json) to change what gets trained:

```json
{
  "steps": "train",
  "epochs": 3,
  "data_root": "/kaggle/input/ai-smart-routing-dataset",
  "batch_size": null,
  "max_steps": null,
  "extra_args": []
}
```

### Loop flags

```
--skip-smoke    Skip the local syntax/smoke check
--skip-push     Skip git commit+push (already pushed)
--smoke-steps   Max steps for local smoke test (default: 20)
--poll-interval Seconds between Kaggle status polls (default: 60)
--timeout       Max wait in seconds (default: 7200)
```

### Kaggle kernel

- **Kernel**: `duckgotsick/ai-smart-routing-train`
- **Script**: [kaggle/notebook.py](kaggle/notebook.py) — clones repo from GitHub, reads run_config.json, tees output to `/kaggle/working/run.log`
- **Metadata**: [kaggle/kernel-metadata.json](kaggle/kernel-metadata.json)
- **Dataset**: `duckgotsick/ai-smart-routing-dataset` — mounted at `/kaggle/input/ai-smart-routing-dataset`

### Git / SSH

SSH key is configured at `/root/.ssh/leducdilac_telegram` (copied from `/home/ducgotsick/.ssh/`). Pushes go to `git@github.com:LeDucDiLac/vietnamese-ai-smart-routing.git`.

## Models

| Model | Backbone | Use |
|---|---|---|
| `vi-router-quality` (teacher) | `microsoft/mdeberta-v3-base` | Offline labeling, evaluation |
| `vi-router-fast` (student) | `microsoft/Multilingual-MiniLM-L12-H384` | Real-time inference (CPU ≤50ms) |

Training uses AMP (`torch.amp.autocast` + `GradScaler`) — model parameters must be cast to fp32 via `.float()` before training or GradScaler will fail with "Attempting to unscale FP16 gradients".

## Dataset

- 50,041 train / 2,778 val / 2,776 test (55,595 total Vietnamese prompts)
- Local copy: `data/processed/` — used for syntax/smoke tests
- Kaggle copy: `/kaggle/input/ai-smart-routing-dataset/`
