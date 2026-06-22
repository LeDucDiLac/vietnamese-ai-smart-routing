# Kaggle Guide

This guide covers the full Kaggle workflow for this repo:

1. prepare the repo locally and push it to GitHub
1. clone the repo inside Kaggle
1. attach the processed dataset
1. install dependencies
1. run training and evaluation
1. optionally run distillation, export, leaderboard refresh, and simulation

`uv` is not required on Kaggle for this workflow. Kaggle runs use plain `python`
and `pip`, driven by `kaggle_run.py`.

## 1. Prepare the repo locally

From your local workspace:

```bash
git status
git add -A
git commit -m "Your message"
git push origin main
```

Make sure the GitHub repo is reachable from Kaggle. If the repo is public, you
can clone it over HTTPS. If it is private, use a Kaggle secret or a public
dataset upload instead of SSH.

## 2. Open Kaggle and enable the right inputs

In the Kaggle notebook:

1. Turn on Internet if you plan to `git clone` from GitHub.
1. Add the repo source as a Kaggle input, or clone it from GitHub in the notebook.
1. Add the processed dataset as a second Kaggle input.

You need a Kaggle dataset that contains `train.jsonl`, `val.jsonl`, and
`test.jsonl`. The repo's training code reads the processed split files from that
mount.

## 3. Clone the repo into Kaggle

Recommended if Internet is enabled:

```bash
!git clone https://github.com/LeDucDiLac/vietnamese-ai-smart-routing.git /kaggle/working/ai-smart-routing
%cd /kaggle/working/ai-smart-routing
```

If you uploaded the repo as a Kaggle dataset instead, copy it into working
storage first:

```bash
!cp -r /kaggle/input/ai-smart-routing /kaggle/working/ai-smart-routing
%cd /kaggle/working/ai-smart-routing
```

## 4. Install dependencies

The repo does not use `uv` inside Kaggle. Use the runner's built-in installer.

For training and evaluation:

```bash
!python kaggle_run.py --install --steps train --data-root /kaggle/input/<processed-dataset-root> --epochs 3
```

That command installs the extra ML dependencies the Kaggle environment may be
missing, then trains the classifier.

If you want to install the package manually instead of using the runner:

```bash
!python -m pip install -U pip
!python -m pip install transformers sentencepiece onnx onnxruntime datasets huggingface-hub
```

## 5. Point the runner at the processed dataset

Replace `<processed-dataset-root>` with the Kaggle input that contains the
processed JSONL files.

Examples:

```bash
/kaggle/input/vn-processed-data
/kaggle/input/ai-smart-routing-data
```

The runner accepts either:

1. a root that contains `train.jsonl`, `val.jsonl`, and `test.jsonl`
1. a root that contains `processed/train.jsonl`, `processed/val.jsonl`, and
   `processed/test.jsonl`

## 6. Train the model

Main training command:

```bash
!python kaggle_run.py --install --steps train --data-root /kaggle/input/<processed-dataset-root> --epochs 3
```

Useful knobs:

```bash
!python kaggle_run.py --install --steps train --data-root /kaggle/input/<processed-dataset-root> --epochs 3 --batch-size 16 --lr 2e-5
!python kaggle_run.py --install --steps train --data-root /kaggle/input/<processed-dataset-root> --epochs 1 --max-steps 20
```

Training outputs go to:

```bash
/kaggle/working/runs/quality
```

That directory contains:

1. `model.pt`
1. `tokenizer/`
1. `meta.json`
1. `history.jsonl`

## 7. Run evaluation

Offline routing evaluation:

```bash
!python kaggle_run.py --steps simulate --data-root /kaggle/input/<processed-dataset-root>
```

The eval step:

1. reads `val.jsonl` if it exists and has rows
1. falls back to `train.jsonl` if validation is empty
1. builds a Vietnamese response cache
1. replays the router policies
1. writes the report to:

```bash
/kaggle/working/runs/eval/report.json
```

If you want to limit the eval size for a quick smoke test:

```bash
!python kaggle_run.py --steps simulate --data-root /kaggle/input/<processed-dataset-root> --sim-limit 50
```

## 8. Optional pipeline stages

The runner supports more than train/eval. Run them in order if you need the full
pipeline.

```bash
!python kaggle_run.py --install --steps synth dataset train distill export leaderboard simulate --data-root /kaggle/input/<processed-dataset-root>
```

Stage summary:

1. `synth` generates synthetic Vietnamese prompts.
1. `dataset` builds `train.jsonl`, `val.jsonl`, and `test.jsonl`.
1. `train` fits `vi-router-quality`.
1. `distill` trains the smaller `vi-router-fast` student.
1. `export` writes the ONNX INT8 artifact.
1. `leaderboard` refreshes the capability leaderboard.
1. `simulate` runs the offline routing eval.

If you already have `data/processed`, you usually only need `train` and
`simulate`.

## 9. Where outputs land

By default on Kaggle:

```bash
/kaggle/working/runs
```

Subdirectories:

1. `quality/` for the trained classifier
1. `fast/` for the distilled student
1. `onnx/` for exported inference artifacts
1. `leaderboard/` for leaderboard refresh outputs
1. `eval/` for the offline evaluation report

## 10. Common pitfalls

1. Do not rely on `uv` inside Kaggle. This repo uses `pip` there.
1. Do not point `--data-root` at the wrong level. It must resolve to the
   processed split files.
1. If `git clone` fails, turn on Kaggle Internet or upload the repo as a Kaggle
   dataset instead.
1. If training cannot find `train.jsonl`, verify that your Kaggle input mount has
   `train.jsonl` or `processed/train.jsonl`.

