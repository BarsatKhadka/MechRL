"""Multiple-choice "anchored bias" on GPT-2 small (Li & Gao, ACL 2025 Findings,
arXiv:2405.03205) -- HELD-OUT candidate, ON PROBATION (run the ceiling probe).

GPT-2 small has a positional bias toward the first option "A" in MCQ prompts,
regardless of content. The circuit is localised: heads L8.H1, L10.H8 + MLP layer 9.

    clean:     "Question: Which is the answer? Answer Choices: (A) {a} (B) {b} Answer: ("  -> "A"
    corrupted: same prompt with the option CONTENTS swapped: "(A) {b} (B) {a}"
    metric: -(logit["A"] - logit["B"]) at the final position.

HONEST CAVEAT: the behaviour is POSITIONAL, not content-driven, so swapping the
option contents may NOT flip the model's output (it still says "A"). If so, KL_cut
is small (weak counterfactual) and the task fails the ceiling gate -- that is the
expected outcome and the probe will SHOW it empirically rather than us asserting it.
If KL_cut comes back healthy, great, it joins the suite. RUN THE PROBE before trusting.
"""

from __future__ import annotations

import random
from typing import Callable

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# Single-token option words (filtered again at build for the leading-space encoding).
_OPTIONS = [
    "cat", "dog", "car", "book", "tree", "house", "road", "river", "table", "chair",
    "apple", "water", "music", "money", "paper", "phone", "glass", "stone", "bread", "cloud",
]
_TEMPLATE = "Question: Which is the answer? Answer Choices: (A) {a} (B) {b} Answer: ("


def _single_tok(tok, s: str, lead: str = " ") -> bool:
    return len(tok(lead + s, add_special_tokens=False)["input_ids"]) == 1


def _letter_id(tok, letter: str):
    # The answer follows "(" with no space, e.g. "...Answer: (A".
    ids = tok(letter, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def _build_mcq_batch(model, batch_size, seed, device):
    tok = model.tokenizer
    rng = random.Random(seed)
    opts = [w for w in _OPTIONS if _single_tok(tok, w)]
    if len(opts) < 4:
        raise RuntimeError("not enough single-token MCQ options")
    a_id, b_id = _letter_id(tok, "A"), _letter_id(tok, "B")
    if a_id is None or b_id is None:
        raise RuntimeError("'A'/'B' not single tokens")

    clean_ids, corrupt_ids, correct, wrong, target_len = [], [], [], [], None
    tries = 0
    while len(clean_ids) < batch_size and tries < batch_size * 200:
        tries += 1
        a, b = rng.sample(opts, 2)
        clean_s = _TEMPLATE.format(a=a, b=b)
        corr_s = _TEMPLATE.format(a=b, b=a)          # swap option CONTENTS, keep positions
        ci = tok(clean_s, add_special_tokens=False)["input_ids"]
        xi = tok(corr_s, add_special_tokens=False)["input_ids"]
        if len(ci) != len(xi):
            continue
        if target_len is None:
            target_len = len(ci)
        if len(ci) != target_len:
            continue
        clean_ids.append(ci); corrupt_ids.append(xi)
        correct.append(a_id); wrong.append(b_id)     # "A" is the (biased) answer we measure
    if len(clean_ids) < batch_size:
        raise RuntimeError(f"only built {len(clean_ids)}/{batch_size} MCQ prompts; add more options")
    return (torch.tensor(clean_ids, dtype=torch.long, device=device),
            torch.tensor(corrupt_ids, dtype=torch.long, device=device),
            torch.tensor(correct, dtype=torch.long, device=device),
            torch.tensor(wrong, dtype=torch.long, device=device))


def _metric(correct: torch.Tensor, wrong: torch.Tensor) -> Callable:
    def m(logits):
        last = logits[:, -1, :]
        n = last.shape[0]
        c, w = correct[:n].to(last.device), wrong[:n].to(last.device)
        idx = torch.arange(n, device=last.device)
        return -(last[idx, c] - last[idx, w]).mean()
    return m


class MCQAnchoredBiasTask(Task):
    """Positional 'A' bias in multiple-choice (heads L8.H1, L10.H8 + MLP9). On probation."""

    name = "mcq_anchored_bias"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        cv, xv, icv, iwv = _build_mcq_batch(model, self.num_examples, self.seed, self.device)
        ct, xt, ict, iwt = _build_mcq_batch(model, self.num_examples, self.seed + 1, self.device)
        self._model = model
        meta = {"source": "Li & Gao 2025 (arXiv:2405.03205)", "task": self.name}
        self._validation = TaskBatch(
            clean_tokens=cv, corrupted_tokens=xv, correct_labels=icv, wrong_labels=iwv,
            metric=_metric(icv, iwv), metadata=meta)
        self._test = TaskBatch(
            clean_tokens=ct, corrupted_tokens=xt, correct_labels=ict, wrong_labels=iwt,
            metric=_metric(ict, iwt), metadata=meta)
