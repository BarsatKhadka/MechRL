"""Successor Heads task (Gould et al. 2024) — held-out evaluation.

Successor heads implement an "increment" function: they take an ordered token
(a day, month, number, letter, etc.) and predict the next one.

Examples:
    "The day after Monday is"   -> " Tuesday"
    "The month after January is" -> " February"
    "The number after seven is"  -> " eight"

We test across three categories (days, months, written numbers) for diversity.
For each example:
    clean prompt:     "The day after Monday is"        -> predict " Tuesday"
    corrupted prompt: "The day after Wednesday is"     -> predict " Thursday"
    metric: -(logit[" Tuesday"] - logit[" Thursday"])  at final position

When the agent's circuit faithfully includes the successor head, the model
on a clean prompt strongly prefers Tuesday over Thursday. When the circuit
omits it, the preference drops.

Held-out test — NOT used for training.

Reference: Gould, Ong, Ogden, Conmy 2024,
"Successor Heads: Recurring, Interpretable Attention Heads In The Wild"
https://arxiv.org/abs/2312.09230
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

import torch
from transformer_lens import HookedTransformer

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# Each category is (template, sequence). The template gets formatted with one
# of the sequence elements; the next element in the sequence is the answer.
_CATEGORIES = [
    (
        "The day after {x} is",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    ),
    (
        "The month after {x} is",
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"],
    ),
    (
        "The number after {x} is",
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    ),
]


def _filter_single_token_sequence(
    tokenizer, sequence: List[str]
) -> List[str]:
    """Keep only items that tokenize as a single token with leading space."""
    out = []
    for item in sequence:
        toks = tokenizer(" " + item, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            out.append(item)
    return out


def _prepare_categories(tokenizer) -> List[Tuple[str, List[str]]]:
    """Filter each category's sequence to keep only items that are valid for
    single-token logit comparison."""
    prepared = []
    for template, seq in _CATEGORIES:
        filt = _filter_single_token_sequence(tokenizer, seq)
        if len(filt) >= 3:  # need at least 3 to have current + correct + wrong
            prepared.append((template, filt))
    return prepared


def _build_successor_batch(
    model: HookedTransformer,
    batch_size: int,
    seed: int,
    device: str,
):
    """Generate clean & corrupted prompts.

    For each example:
      - Pick a category and a starting index i in its sequence (with i < len-1).
      - Clean: template formatted with sequence[i], correct answer = sequence[i+1]
      - Corrupted: template formatted with sequence[j], j chosen so j+1 != i+1
      - Wrong-label (for logit-diff metric) = sequence[j+1] (corrupted's correct)
    """
    tokenizer = model.tokenizer
    rng = random.Random(seed)
    categories = _prepare_categories(tokenizer)

    if not categories:
        raise RuntimeError("No category has enough single-token elements.")

    clean_strs, corrupt_strs, correct_ids, wrong_ids = [], [], [], []
    for _ in range(batch_size):
        template, seq = rng.choice(categories)
        # Pick i so that sequence[i+1] exists
        i = rng.randrange(0, len(seq) - 1)
        # Pick j != i and j < len-1 so sequence[j+1] exists; also require
        # sequence[j+1] != sequence[i+1] so distractor is meaningful.
        valid_j = [
            k for k in range(len(seq) - 1)
            if k != i and seq[k + 1] != seq[i + 1]
        ]
        if not valid_j:
            # very small sequence; skip this example by retrying with new category
            continue
        j = rng.choice(valid_j)

        clean_strs.append(template.format(x=seq[i]))
        corrupt_strs.append(template.format(x=seq[j]))

        correct = seq[i + 1]
        wrong = seq[j + 1]
        correct_id = tokenizer(" " + correct, add_special_tokens=False)["input_ids"][0]
        wrong_id = tokenizer(" " + wrong, add_special_tokens=False)["input_ids"][0]
        correct_ids.append(correct_id)
        wrong_ids.append(wrong_id)

    # All prompts in the same category have the same length, but across
    # categories lengths differ. Pad to max length.
    clean_lists = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in clean_strs]
    corrupt_lists = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in corrupt_strs]
    max_len = max(max(len(l) for l in clean_lists), max(len(l) for l in corrupt_lists))
    pad = tokenizer.eos_token_id

    # Left-pad so the final position (where the metric reads) is always the same
    # logical token across all examples regardless of original length.
    def lpad(l):
        return [pad] * (max_len - len(l)) + l

    clean_lists = [lpad(l) for l in clean_lists]
    corrupt_lists = [lpad(l) for l in corrupt_lists]

    clean_tokens = torch.tensor(clean_lists, dtype=torch.long, device=device)
    corrupt_tokens = torch.tensor(corrupt_lists, dtype=torch.long, device=device)
    correct_t = torch.tensor(correct_ids, dtype=torch.long, device=device)
    wrong_t = torch.tensor(wrong_ids, dtype=torch.long, device=device)
    return clean_tokens, corrupt_tokens, correct_t, wrong_t


def _build_logit_diff_metric(correct: torch.Tensor, wrong: torch.Tensor) -> Callable:
    """Metric: -(logit[correct] - logit[wrong]) at final position. Lower = better."""

    def metric(logits: torch.Tensor) -> torch.Tensor:
        last_logits = logits[:, -1, :]
        n = last_logits.shape[0]
        c = correct[:n].to(last_logits.device)
        w = wrong[:n].to(last_logits.device)
        idx = torch.arange(n, device=last_logits.device)
        return -(last_logits[idx, c] - last_logits[idx, w]).mean()

    return metric


class SuccessorHeadsTask(Task):
    """Held-out test task for successor head behavior."""

    name = "successor_heads"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)

        clean_v, corrupt_v, correct_v, wrong_v = _build_successor_batch(
            model, self.num_examples, seed=self.seed, device=self.device
        )
        clean_t, corrupt_t, correct_t, wrong_t = _build_successor_batch(
            model, self.num_examples, seed=self.seed + 1, device=self.device
        )

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=clean_v,
            corrupted_tokens=corrupt_v,
            correct_labels=correct_v,
            wrong_labels=wrong_v,
            metric=_build_logit_diff_metric(correct_v, wrong_v),
            metadata={
                "source": "Gould et al. 2024 (synthetic templated version)",
                "categories": ["days", "months", "numbers"],
                "task": "predict next item in ordered sequence",
            },
        )
        self._test = TaskBatch(
            clean_tokens=clean_t,
            corrupted_tokens=corrupt_t,
            correct_labels=correct_t,
            wrong_labels=wrong_t,
            metric=_build_logit_diff_metric(correct_t, wrong_t),
            metadata={
                "source": "Gould et al. 2024 (synthetic templated version)",
                "categories": ["days", "months", "numbers"],
                "task": "predict next item in ordered sequence",
            },
        )
