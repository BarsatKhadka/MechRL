"""Copy Suppression task (McDougall et al. 2023) — held-out evaluation.

Tests whether the agent's discovered circuit reproduces L10.H7's suppression
behavior: when a token T appears earlier in context, the model partially
SUPPRESSES predicting T again (instead of dumbly repeating).

Templated prompts of the form:
    "{name1} and {name2} are friends. {name1} was eating an {food}.
     {name2} was also eating an"

The natural continuation is " {food}" — both pattern-completion and induction
push for it. L10.H7 partially suppresses " {food}" because it appeared recently.

Clean prompt:     "{food}" appears earlier (suppression active)
Corrupted prompt: a different food appears earlier (suppression target differs)

Metric: logit of "{food}" at the final position. Negated so lower = better
(matches ACDC convention — when the agent's circuit preserves L10.H7's
suppression, the logit of {food} stays low compared to a circuit that
removed L10.H7).

WARNING: copy suppression is a SMALLER signal than IOI / docstring / induction.
L10.H7's typical contribution is ~0.5-1.0 logits, not 3+. Expect smaller clean
vs corrupted gaps.

Not used in training — held out as the transfer evaluation.
"""

from __future__ import annotations

import random
from typing import Callable, List

import torch
from transformer_lens import HookedTransformer

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


NAMES = [
    "John", "Mary", "James", "Sarah", "Michael", "Emily", "David", "Anna",
    "Robert", "Laura", "Daniel", "Olivia", "Henry", "Sophia", "Thomas",
    "Emma", "Charles", "Alice", "Edward", "Grace",
]

FOODS = [
    "sandwich", "pizza", "burger", "cookie", "muffin", "donut",
    "salad", "steak", "pancake", "taco", "wrap", "biscuit",
]


def _filter_single_token(tokenizer, words: List[str]) -> List[str]:
    """Keep only words that tokenize as a single token with a leading space."""
    out = []
    for w in words:
        toks = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            out.append(w)
    return out


def _build_cs_prompt(name1: str, name2: str, food: str) -> str:
    """Prompt template — ends with ' a' so the model predicts the next word.
    All foods start with consonants so 'a {food}' is grammatical English.
    """
    return (
        f"{name1} and {name2} are friends. "
        f"{name1} was eating a {food}. "
        f"{name2} was also eating a"
    )


def _build_cs_batch(
    model: HookedTransformer,
    batch_size: int,
    names: List[str],
    foods: List[str],
    seed: int,
    device: str,
):
    """Generate clean and corrupted prompts, plus the food token id (the answer
    whose logit we measure suppression of)."""
    tokenizer = model.tokenizer
    rng = random.Random(seed)

    clean_strs, corrupt_strs, food_ids = [], [], []
    for _ in range(batch_size):
        name1, name2 = rng.sample(names, 2)
        # Two different foods so corrupted prompt is structurally identical
        # but the EARLIER food differs from the one we're measuring.
        food_clean, food_other = rng.sample(foods, 2)

        # Clean: the food we measure ("food_clean") appears earlier in the prompt
        clean_strs.append(_build_cs_prompt(name1, name2, food_clean))

        # Corrupted: the prompt mentions a DIFFERENT food earlier. The metric
        # still measures logit of "food_clean" — which is NOT in early context
        # in this version, so L10.H7 has nothing to suppress.
        corrupt_strs.append(_build_cs_prompt(name1, name2, food_other))

        # Token id of food_clean (with leading space) — what we measure
        food_id = tokenizer(" " + food_clean, add_special_tokens=False)["input_ids"][0]
        food_ids.append(food_id)

    # Tokenize prompts. Length varies with names/foods chosen. We LEFT-pad
    # with EOS so the meaningful prediction position is always the LAST token.
    clean_lists = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in clean_strs]
    corrupt_lists = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in corrupt_strs]

    max_len = max(max(len(l) for l in clean_lists), max(len(l) for l in corrupt_lists))
    pad = tokenizer.eos_token_id

    def lpad(l):
        return [pad] * (max_len - len(l)) + l

    clean_lists = [lpad(l) for l in clean_lists]
    corrupt_lists = [lpad(l) for l in corrupt_lists]

    clean_tokens = torch.tensor(clean_lists, dtype=torch.long, device=device)
    corrupt_tokens = torch.tensor(corrupt_lists, dtype=torch.long, device=device)
    food_ids_t = torch.tensor(food_ids, dtype=torch.long, device=device)
    return clean_tokens, corrupt_tokens, food_ids_t


def _build_cs_metric(food_ids: torch.Tensor) -> Callable:
    """Metric: mean logit of the food token at the final position.

    LOWER = better (matches ACDC convention).
    - Clean: food appears earlier → L10.H7 suppresses → low logit → good
    - Corrupted: food NOT in early context → no suppression → high logit
    """

    def metric(logits: torch.Tensor) -> torch.Tensor:
        last_logits = logits[:, -1, :]
        n = last_logits.shape[0]
        ids = food_ids[:n].to(last_logits.device)
        idx = torch.arange(n, device=last_logits.device)
        food_logit = last_logits[idx, ids]
        return food_logit.mean()

    return metric


class CopySuppressionTask(Task):
    """Held-out test task. NOT used for training."""

    name = "copy_suppression"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        tokenizer = model.tokenizer

        names = _filter_single_token(tokenizer, NAMES)
        foods = _filter_single_token(tokenizer, FOODS)
        if len(foods) < 5 or len(names) < 5:
            raise RuntimeError(f"Need more single-token names/foods (have {len(names)}/{len(foods)}).")

        clean_v, corrupt_v, food_v = _build_cs_batch(
            model, self.num_examples, names, foods, seed=self.seed, device=self.device
        )
        clean_t, corrupt_t, food_t = _build_cs_batch(
            model, self.num_examples, names, foods, seed=self.seed + 1, device=self.device
        )

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=clean_v,
            corrupted_tokens=corrupt_v,
            correct_labels=food_v,
            wrong_labels=None,
            metric=_build_cs_metric(food_v),
            metadata={
                "source": "McDougall et al. 2023 (synthetic templated version)",
                "canonical_head": "L10.H7",
                "task": "measure suppression of recently-mentioned token",
                "n_names": len(names),
                "n_foods": len(foods),
            },
        )
        self._test = TaskBatch(
            clean_tokens=clean_t,
            corrupted_tokens=corrupt_t,
            correct_labels=food_t,
            wrong_labels=None,
            metric=_build_cs_metric(food_t),
            metadata={
                "source": "McDougall et al. 2023 (synthetic templated version)",
                "canonical_head": "L10.H7",
                "task": "measure suppression of recently-mentioned token",
                "n_names": len(names),
                "n_foods": len(foods),
            },
        )
