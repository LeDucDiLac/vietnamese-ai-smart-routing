# Kaggle Usage

Run training and evaluation on Kaggle only.

1. Copy the repo from `/kaggle/input` to writable storage and switch into it:

```bash
!cp -r /kaggle/input/ai-smart-routing /kaggle/working/repo
%cd /kaggle/working/repo
```

2. Train the classifier on the processed dataset mounted in Kaggle:

```bash
!python kaggle_run.py --install --steps train --data-root /kaggle/input/<processed-dataset-root> --epochs 3
```

3. Run offline evaluation on the same Kaggle dataset:

```bash
!python kaggle_run.py --steps simulate --data-root /kaggle/input/<processed-dataset-root>
```

`<processed-dataset-root>` must point to the Kaggle dataset that contains `train.jsonl`, `val.jsonl`, and `test.jsonl` under `data/processed/` or directly at the mounted root if those files are at the top level.
