"""Dump real samples from every task to disk so you can inspect them.

Writes two files per task into datasets_dump/:
  - {task}_examples.md   — human-readable side-by-side clean vs corrupted prompts
  - {task}_summary.json  — machine-readable stats (metric values, token counts)

Run from repo root:
    venv\\Scripts\\python.exe scripts\\dump_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mechrl.tasks import (
    IOITask,
    GreaterThanTask,
    InductionTask,
    DocstringTask,
    DocstringGPT2Task,
    CopySuppressionTask,
    SuccessorHeadsTask,
    TracrReverseTask,
    TracrProportionTask,
)


DUMP_DIR = Path(__file__).resolve().parents[1] / "datasets_dump"
DUMP_DIR.mkdir(exist_ok=True)


def _safe_decode(model, tokens):
    try:
        return model.to_string(tokens)
    except (AssertionError, AttributeError):
        return "tokens=" + str(tokens.tolist())


def _describe_diff(clean_tokens, corrupt_tokens, model):
    """Summarize which positions differ between clean and corrupted.

    Returns a short string like "2 token(s) differ at position(s) 5, 11:
    clean='Mary,John' vs corrupted='Alice,Bob'".
    """
    if clean_tokens.shape != corrupt_tokens.shape:
        return f"SHAPE MISMATCH: clean {tuple(clean_tokens.shape)} vs corrupted {tuple(corrupt_tokens.shape)}"

    diff_positions = (clean_tokens != corrupt_tokens).nonzero().flatten().tolist()
    if not diff_positions:
        return "**WARNING: clean and corrupted are IDENTICAL — bug**"

    n_diff = len(diff_positions)
    total = clean_tokens.numel()
    try:
        clean_diff_tokens = [model.tokenizer.decode([clean_tokens[p].item()]).strip() for p in diff_positions[:5]]
        corrupt_diff_tokens = [model.tokenizer.decode([corrupt_tokens[p].item()]).strip() for p in diff_positions[:5]]
        more = f" (+{n_diff - 5} more)" if n_diff > 5 else ""
        pos_str = ", ".join(str(p) for p in diff_positions[:5]) + more
        c_str = "|".join(repr(t) for t in clean_diff_tokens)
        x_str = "|".join(repr(t) for t in corrupt_diff_tokens)
        return f"{n_diff}/{total} tokens differ at positions {pos_str}; clean={c_str} vs corrupted={x_str}"
    except (AssertionError, AttributeError):
        return f"{n_diff}/{total} tokens differ at positions {diff_positions[:10]}"


def dump_task(task, n_examples_to_show: int = 20):
    print(f"\n{'='*70}\nDumping task: {task.name}\n{'='*70}")

    batch = task.validation_batch()
    model = task.model

    # Sanity: both clean and corrupted must exist and have the same shape
    assert batch.clean_tokens.shape == batch.corrupted_tokens.shape, (
        f"shape mismatch: clean {batch.clean_tokens.shape} "
        f"vs corrupted {batch.corrupted_tokens.shape}"
    )
    has_corrupted = batch.corrupted_tokens is not None
    n_show = min(n_examples_to_show, batch.batch_size)

    # Compute clean/corrupted metric values per-prompt
    with torch.no_grad():
        clean_logits = model(batch.clean_tokens)
        corrupt_logits = model(batch.corrupted_tokens)

    # Per-prompt metric is unreliable for most ACDC-derived metrics because
    # the labels were partialled in at construction time for the full batch.
    # Passing logits[i:i+1] doesn't re-slice the labels, so the result mixes
    # one prompt's logits with all the batch's labels. We skip per-prompt
    # display and only report aggregates, which ARE correct.
    per_prompt_clean = []
    per_prompt_corrupt = []
    per_prompt_supported = False

    # Aggregate metrics
    full_clean = batch.metric(clean_logits)
    full_corrupt = batch.metric(corrupt_logits)
    clean_agg = full_clean.item() if torch.is_tensor(full_clean) and full_clean.dim() == 0 else float(full_clean.mean())
    corrupt_agg = full_corrupt.item() if torch.is_tensor(full_corrupt) and full_corrupt.dim() == 0 else float(full_corrupt.mean())

    # Build the markdown view
    md_lines = [
        f"# Task: {task.name}",
        "",
        f"- **Model**: {model.cfg.model_name} (n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads})",
        f"- **Batch size**: {batch.batch_size}",
        f"- **Sequence length**: {batch.seq_len}",
        f"- **Has corrupted prompts**: {has_corrupted}",
        f"- **Metric aggregate (clean)**: {clean_agg:.4f}",
        f"- **Metric aggregate (corrupted)**: {corrupt_agg:.4f}",
        "",
        "Metadata:",
        "```json",
        json.dumps(batch.metadata, indent=2),
        "```",
        "",
        "## Examples",
        "",
        f"Showing first {n_show} examples. Note that for IOI the metric is negated logit-diff"
        " (lower = better, model prefers correct). For greater-than the metric is negated"
        " probability difference between valid-year and invalid-year continuations.",
        "",
    ]

    for i in range(n_show):
        try:
            clean_str = model.to_string(batch.clean_tokens[i])
            corrupt_str = model.to_string(batch.corrupted_tokens[i])
        except (AssertionError, AttributeError):
            # No tokenizer (e.g. tracr-compiled model) — show raw token ids
            clean_str = "tokens=" + str(batch.clean_tokens[i].tolist())
            corrupt_str = "tokens=" + str(batch.corrupted_tokens[i].tolist())

        # Highlight where clean and corrupted differ (ABC patching design:
        # they SHOULD be almost identical with just a few token differences).
        diff_summary = _describe_diff(batch.clean_tokens[i], batch.corrupted_tokens[i], model)
        md_lines.append(f"### Example {i}")
        md_lines.append("")
        md_lines.append(f"- **CLEAN**:     `{clean_str}`")
        md_lines.append(f"- **CORRUPTED**: `{corrupt_str}`")
        if diff_summary:
            md_lines.append(f"- **Difference**: {diff_summary}")

        if batch.correct_labels is not None and batch.correct_labels.dim() == 1:
            label = batch.correct_labels[i].item()
            # For IOI: label is a token id. For greater-than: label is YY threshold int.
            if task.name == "ioi":
                md_lines.append(f"- **Correct answer token**: `{label}` -> `{model.to_string([label])!r}`")
            elif task.name == "greaterthan":
                md_lines.append(
                    f"- **Year threshold (YY)**: `{label}` — model should put probability on YY > {label:02d}"
                )

        if per_prompt_supported:
            md_lines.append(f"- **Metric clean**: {per_prompt_clean[i]:.4f}")
            md_lines.append(f"- **Metric corrupted**: {per_prompt_corrupt[i]:.4f}")
        md_lines.append("")

    md_path = DUMP_DIR / f"{task.name}_examples.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  wrote {md_path.relative_to(DUMP_DIR.parent)}")

    # Build the JSON summary
    summary = {
        "task": task.name,
        "model": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "batch_size": int(batch.batch_size),
        "seq_len": int(batch.seq_len),
        "has_corrupted": has_corrupted,
        "metric_aggregate_clean": clean_agg,
        "metric_aggregate_corrupted": corrupt_agg,
        "per_prompt_metric_clean": per_prompt_clean,
        "per_prompt_metric_corrupted": per_prompt_corrupt,
        "metadata": batch.metadata,
        "first_5_clean_prompts": [_safe_decode(model, batch.clean_tokens[i]) for i in range(min(5, n_show))],
        "first_5_corrupted_prompts": [_safe_decode(model, batch.corrupted_tokens[i]) for i in range(min(5, n_show))],
    }
    json_path = DUMP_DIR / f"{task.name}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {json_path.relative_to(DUMP_DIR.parent)}")

    # Compact stdout summary
    print(f"  shapes: clean={tuple(batch.clean_tokens.shape)} corrupted={tuple(batch.corrupted_tokens.shape)}")
    print(f"  clean aggregate metric:     {clean_agg:+.4f}")
    print(f"  corrupted aggregate metric: {corrupt_agg:+.4f}")
    if per_prompt_supported:
        print(f"  per-prompt clean range:     [{min(per_prompt_clean):+.3f}, {max(per_prompt_clean):+.3f}]")
        print(f"  per-prompt corrupted range: [{min(per_prompt_corrupt):+.3f}, {max(per_prompt_corrupt):+.3f}]")


def main():
    print(f"Dumping datasets to {DUMP_DIR}/")
    dump_task(IOITask(num_examples=20, device="cpu"))
    dump_task(GreaterThanTask(num_examples=20, device="cpu"))
    dump_task(InductionTask(num_examples=20, half_len=25, device="cpu"))

    for cls, kwargs in [
        (DocstringGPT2Task, {"num_examples": 20}),
        (CopySuppressionTask, {"num_examples": 20}),
        (SuccessorHeadsTask, {"num_examples": 20}),
        (DocstringTask, {"num_examples": 10}),
        (TracrReverseTask, {}),
        (TracrProportionTask, {"num_examples": 20}),
    ]:
        try:
            dump_task(cls(device="cpu", **kwargs))
        except Exception as e:
            msg = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"\n[{cls.__name__}] SKIPPED: {type(e).__name__}: {msg}")

    print("\nDone. Open the .md files in datasets_dump/ to inspect side by side.")


if __name__ == "__main__":
    main()
