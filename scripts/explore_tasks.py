"""Build each task and show what its data looks like.

For each task we print:
- shapes of clean and corrupted token tensors
- the first few prompts decoded (clean vs corrupted side by side)
- the answer tokens
- the metric value on the clean batch (gold standard) and corrupted batch (baseline)

Run from repo root:
    venv\\Scripts\\python.exe scripts\\explore_tasks.py
"""

from __future__ import annotations

import torch

from mechrl.tasks import IOITask, GreaterThanTask


def show_task(task, decode_first: int = 4):
    print(f"\n{'=' * 70}")
    print(f"TASK: {task.name}")
    print(f"{'=' * 70}")

    batch = task.validation_batch()
    model = task.model

    print(f"  model:                  {model.cfg.model_name} "
          f"(n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads})")
    print(f"  clean_tokens shape:     {tuple(batch.clean_tokens.shape)}")
    print(f"  corrupted_tokens shape: {tuple(batch.corrupted_tokens.shape)}")
    if batch.correct_labels is not None:
        print(f"  correct_labels shape:   {tuple(batch.correct_labels.shape)}")
    print(f"  metadata:               {batch.metadata}")

    print(f"\n  first {decode_first} prompts (clean vs corrupted):")
    for i in range(min(decode_first, batch.batch_size)):
        clean_str = model.to_string(batch.clean_tokens[i])
        corrupt_str = model.to_string(batch.corrupted_tokens[i])
        if len(clean_str) > 200:
            clean_str = clean_str[:200] + "..."
            corrupt_str = corrupt_str[:200] + "..."
        print(f"    [{i}] CLEAN:     {clean_str!r}")
        print(f"        CORRUPTED: {corrupt_str!r}")
        if batch.correct_labels is not None and batch.correct_labels.dim() == 1:
            label_id = batch.correct_labels[i].item()
            print(f"        correct_label: {label_id} -> {model.to_string([label_id])!r}")

    with torch.no_grad():
        clean_logits = model(batch.clean_tokens)
        corrupt_logits = model(batch.corrupted_tokens)

    clean_score = batch.metric(clean_logits)
    corrupt_score = batch.metric(corrupt_logits)

    def fmt(x):
        if torch.is_tensor(x):
            if x.dim() == 0:
                return f"{x.item():.4f}"
            return f"mean={x.mean().item():.4f}  std={x.std().item():.4f}  shape={tuple(x.shape)}"
        return str(x)

    print(f"\n  metric on CLEAN batch:     {fmt(clean_score)}")
    print(f"  metric on CORRUPTED batch: {fmt(corrupt_score)}")
    print(f"  (For IOI: clean ~ -3.5 means logit-diff +3.5 in favor of correct answer.")
    print(f"   ACDC's convention negates logit-diff to act as a loss — lower is better.)")


def main():
    print("Building IOI task...")
    show_task(IOITask(num_examples=8, device="cpu"))

    print("\nBuilding Greater-than task...")
    show_task(GreaterThanTask(num_examples=8, device="cpu"))

    print("\nDone.")


if __name__ == "__main__":
    main()
