"""How well does the full GPT-2 model do on our induction task?

Computes multiple metrics on the validation batch:
  - Mean log P(correct) — what we use as the task metric
  - Mean P(correct) — derived probability
  - Top-1 accuracy — model's #1 prediction matches correct
  - Top-5 accuracy — correct is in model's top 5 predictions
  - Per-prompt P(correct) distribution

This tells us if the dataset is "easy enough" that the model genuinely
demonstrates induction behavior on most prompts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mechrl.tasks import InductionTask


def main():
    task = InductionTask(num_examples=50, half_len=8, device="cpu")
    model = task.model
    batch = task.validation_batch()

    print(f"Task: induction (Olsson-style, random tokens, half_len={task.half_len})")
    print(f"Sequence length: {batch.seq_len}")
    print(f"Batch size: {batch.batch_size}")
    print()

    with torch.no_grad():
        logits = model(batch.clean_tokens)

    last_logits = logits[:, -1, :]  # [batch, vocab]
    probs = F.softmax(last_logits, dim=-1)
    log_probs = F.log_softmax(last_logits, dim=-1)

    labels = batch.correct_labels  # [batch]
    n = batch.batch_size
    idx = torch.arange(n)

    # Probability of correct token per prompt
    p_correct = probs[idx, labels]  # [batch]
    log_p_correct = log_probs[idx, labels]  # [batch]

    # Top-1 prediction
    top1 = last_logits.argmax(dim=-1)
    top1_correct = (top1 == labels).float()

    # Top-5 predictions
    top5_ids = last_logits.topk(5, dim=-1).indices  # [batch, 5]
    top5_correct = (top5_ids == labels.unsqueeze(1)).any(dim=-1).float()

    print(f"  Mean log P(correct):  {log_p_correct.mean().item():+.4f}")
    print(f"  Mean P(correct):      {p_correct.mean().item():.2%}")
    print(f"  Top-1 accuracy:       {top1_correct.mean().item():.2%}")
    print(f"  Top-5 accuracy:       {top5_correct.mean().item():.2%}")
    print()

    print(f"Per-prompt P(correct) distribution:")
    p_sorted = p_correct.sort(descending=True).values
    print(f"  best:    {p_sorted[0].item():.2%}")
    print(f"  median:  {p_sorted[n // 2].item():.2%}")
    print(f"  worst:   {p_sorted[-1].item():.2%}")
    print(f"  >90%:    {(p_correct > 0.9).sum().item()}/{n} prompts")
    print(f"  >50%:    {(p_correct > 0.5).sum().item()}/{n} prompts")
    print(f"  >10%:    {(p_correct > 0.1).sum().item()}/{n} prompts")
    print(f"  <1%:     {(p_correct < 0.01).sum().item()}/{n} prompts")

    print()
    print(f"Baseline for reference: random guess = {1/50257:.6%}")
    print(f"Model is {p_correct.mean().item() / (1/50257):.0f}x better than random.")

    # Show a few example predictions
    print("\nExample prompts (first 5):")
    tokenizer = model.tokenizer
    for i in range(min(5, n)):
        prompt_str = model.to_string(batch.clean_tokens[i])
        correct_tok = tokenizer.decode([labels[i].item()])
        predicted_tok = tokenizer.decode([top1[i].item()])
        p = p_correct[i].item()
        match = "OK" if top1[i] == labels[i] else "X "
        # Truncate long prompts
        if len(prompt_str) > 80:
            prompt_str = "..." + prompt_str[-77:]
        print(f"  [{match}] prompt: {prompt_str!r}")
        print(f"       expected: {correct_tok!r}   predicted: {predicted_tok!r}   P(correct)={p:.2%}")


if __name__ == "__main__":
    main()
