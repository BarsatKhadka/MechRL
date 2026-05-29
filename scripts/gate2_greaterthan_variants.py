"""GATE 2: Test top-3000 EAP-IG faithfulness for each greater-than variant.

Builds a temporary Task subclass for each variant, runs prefilter, measures
faithfulness at K=3000. Variants that pass (>60%) are good training tasks.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import (
    GREATERTHAN_NOUNS,
    build_year_metric,
    get_valid_years,
    load_gpt2_small,
)
from mechrl.env import build_graph, Prefilter, AblationEngine


TEMPLATES = {
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


def build_batch_for_template(model, clean_fn, bad_fn, n_examples, seed):
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
        return None
    target = lens.most_common(1)[0][0]
    kept = [(g, b, y) for g, b, y in zip(good_l, bad_l, kept_years) if len(g) == target and len(b) == target]
    if len(kept) < n_examples:
        return None
    kept = kept[:n_examples]

    return (
        torch.tensor([g for g, _, _ in kept], dtype=torch.long),
        torch.tensor([b for _, b, _ in kept], dtype=torch.long),
        torch.tensor([y % 100 for _, _, y in kept], dtype=torch.long),
    )


class _GreaterThanVariantTask(Task):
    """Minimal Task subclass for testing a single template variant."""
    def __init__(self, variant_name, clean_fn, bad_fn, num_examples=30, device="cpu", seed=0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)
        self.name = f"greaterthan_{variant_name}"
        self._clean_fn = clean_fn
        self._bad_fn = bad_fn

    def _build(self):
        model = load_gpt2_small(device=self.device)
        good_v, bad_v, yy_v = build_batch_for_template(
            model, self._clean_fn, self._bad_fn, self.num_examples, seed=self.seed
        )
        good_t, bad_t, yy_t = build_batch_for_template(
            model, self._clean_fn, self._bad_fn, self.num_examples, seed=self.seed + 1
        )
        self._model = model
        self._validation = TaskBatch(
            clean_tokens=good_v.to(self.device),
            corrupted_tokens=bad_v.to(self.device),
            correct_labels=yy_v.to(self.device),
            wrong_labels=None,
            metric=build_year_metric(model.tokenizer, yy_v),
            metadata={"variant": self.name},
        )
        self._test = TaskBatch(
            clean_tokens=good_t.to(self.device),
            corrupted_tokens=bad_t.to(self.device),
            correct_labels=yy_t.to(self.device),
            wrong_labels=None,
            metric=build_year_metric(model.tokenizer, yy_t),
            metadata={"variant": self.name},
        )


def main():
    print(f"{'variant':>25} | {'full':>7} | {'cut':>7} | {'top-3000 faithfulness':>22} | gate 2")
    print(f"{'-'*25} | {'-'*7} | {'-'*7} | {'-'*22} | {'-'*8}")
    for tname, (clean_fn, bad_fn) in TEMPLATES.items():
        try:
            task = _GreaterThanVariantTask(tname, clean_fn, bad_fn, num_examples=30, device="cpu")
            graph = build_graph(task.model)
            engine = AblationEngine(task, graph)
            pref = Prefilter(task, graph, ig_steps=5)
            pref.compute(batch_size=10)
            full = engine.full_baseline()
            cut = engine.corrupted_baseline()
            f = engine.faithfulness(pref.candidate_mask(3000))
            passes = "PASS" if f > 0.6 else "FAIL"
            print(f"{tname:>25} | {full:>+7.3f} | {cut:>+7.3f} | {f:>21.2%}  | {passes:>8}")
        except Exception as e:
            msg = str(e).encode("ascii", errors="replace").decode("ascii")[:30]
            print(f"{tname:>25} | ERROR: {msg}")


if __name__ == "__main__":
    main()
