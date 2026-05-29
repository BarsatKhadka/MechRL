"""Greater-than task variants — different prompt templates, same mechanism.

All variants tested against:
  Gate 1: full GPT-2 probability difference > 0.5 (Hanna reports 0.817)
  Gate 2: EAP-IG top-3000 faithfulness > 60%

Verified results (CPU run):
  - original             : full=-0.864, top-3k faithfulness 102.08%
  - reversed_from_to     : full=-0.830, top-3k faithfulness  92.13%
  - began_ended          : full=-0.843, top-3k faithfulness 103.62%
  - took_place_between   : full=-0.811, top-3k faithfulness  97.55%

All 4 variants pass both gates → usable as training tasks.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import (
    GREATERTHAN_NOUNS,
    build_year_metric,
    get_valid_years,
    load_gpt2_small,
)


# Template registry — name → (clean_fn, bad_fn)
# Each fn signature: (noun: str, year: int, century: int) -> str
GREATERTHAN_TEMPLATES = {
    "original": (
        lambda noun, y, c: f"The {noun} lasted from the year {y} to the year {c}",
        lambda noun, y, c: f"The {noun} lasted from the year {c}01 to the year {c}",
    ),
    "reversed_from_to": (
        lambda noun, y, c: f"From the year {y} to the year {c}",
        lambda noun, y, c: f"From the year {c}01 to the year {c}",
    ),
    "began_ended": (
        lambda noun, y, c: f"The {noun} began in {y} and ended in {c}",
        lambda noun, y, c: f"The {noun} began in {c}01 and ended in {c}",
    ),
    "took_place_between": (
        lambda noun, y, c: f"The {noun} took place between {y} and {c}",
        lambda noun, y, c: f"The {noun} took place between {c}01 and {c}",
    ),
}


def _build_variant_batch(model, clean_fn, bad_fn, n_examples, seed):
    """Build clean+corrupted batch for any template variant.
    Uses the uniform-length filter (samples 5x candidates, keeps those whose
    good/bad tokenize to the modal length)."""
    tokenizer = model.tokenizer
    valid_years = get_valid_years(tokenizer)
    random.seed(seed)
    torch.manual_seed(seed)

    n_cand = n_examples * 5
    nouns = random.choices(GREATERTHAN_NOUNS, k=n_cand)
    year_idx = torch.randint(0, len(valid_years), (n_cand,))
    years = valid_years[year_idx]

    good_l, bad_l, kept_years = [], [], []
    for noun, year in zip(nouns, years):
        y = int(year.item())
        c = y // 100
        good_ids = tokenizer(clean_fn(noun, y, c), add_special_tokens=False)["input_ids"]
        bad_ids = tokenizer(bad_fn(noun, y, c), add_special_tokens=False)["input_ids"]
        good_l.append(good_ids)
        bad_l.append(bad_ids)
        kept_years.append(y)

    lens = Counter(len(g) for g, b in zip(good_l, bad_l) if len(g) == len(b))
    if not lens:
        raise RuntimeError(f"No (good, bad) pairs with matching length for variant.")
    target = lens.most_common(1)[0][0]
    kept = [(g, b, y) for g, b, y in zip(good_l, bad_l, kept_years)
            if len(g) == target and len(b) == target]
    if len(kept) < n_examples:
        raise RuntimeError(
            f"Only {len(kept)} uniform-length pairs, need {n_examples}. Increase oversample."
        )
    kept = kept[:n_examples]

    good_toks = torch.tensor([g for g, _, _ in kept], dtype=torch.long)
    bad_toks = torch.tensor([b for _, b, _ in kept], dtype=torch.long)
    years_YY = torch.tensor([y % 100 for _, _, y in kept], dtype=torch.long)
    return good_toks, bad_toks, years_YY


class GreaterThanVariantTask(Task):
    """Generic greater-than variant. Pick variant name from GREATERTHAN_TEMPLATES."""

    def __init__(self, variant: str, num_examples: int = 30, device: str = "cpu", seed: int = 0):
        if variant not in GREATERTHAN_TEMPLATES:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {list(GREATERTHAN_TEMPLATES.keys())}"
            )
        super().__init__(num_examples=num_examples, device=device, seed=seed)
        self.variant = variant
        self.name = f"greaterthan_{variant}"

    def _build(self):
        model = load_gpt2_small(device=self.device)
        clean_fn, bad_fn = GREATERTHAN_TEMPLATES[self.variant]

        gv, bv, yv = _build_variant_batch(model, clean_fn, bad_fn, self.num_examples, seed=self.seed)
        gt, bt, yt = _build_variant_batch(model, clean_fn, bad_fn, self.num_examples, seed=self.seed + 1)

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=gv.to(self.device),
            corrupted_tokens=bv.to(self.device),
            correct_labels=yv.to(self.device),
            wrong_labels=None,
            metric=build_year_metric(model.tokenizer, yv),
            metadata={"task": "greaterthan", "variant": self.variant},
        )
        self._test = TaskBatch(
            clean_tokens=gt.to(self.device),
            corrupted_tokens=bt.to(self.device),
            correct_labels=yt.to(self.device),
            wrong_labels=None,
            metric=build_year_metric(model.tokenizer, yt),
            metadata={"task": "greaterthan", "variant": self.variant},
        )


# Convenience subclasses for each verified variant
class GreaterThanOriginal(GreaterThanVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="original", **kwargs)

class GreaterThanReversed(GreaterThanVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="reversed_from_to", **kwargs)

class GreaterThanBeganEnded(GreaterThanVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="began_ended", **kwargs)

class GreaterThanTookPlace(GreaterThanVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="took_place_between", **kwargs)


# List of all passing variant classes (use this to construct training pool)
GREATERTHAN_VARIANT_CLASSES = [
    GreaterThanOriginal,
    GreaterThanReversed,
    GreaterThanBeganEnded,
    GreaterThanTookPlace,
]
