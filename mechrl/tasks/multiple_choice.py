"""Multiple-choice "anchored bias" on GPT-2 small (Li & Gao, ACL 2025 Findings,
arXiv:2405.03205) -- HELD-OUT candidate. Re-gate with the probe before trusting.

GPT-2 small has a positional bias toward the FIRST option's label in MCQ prompts.
The circuit is localised: heads L8.H1, L10.H8 + MLP layer 9.

The behaviour is POSITIONAL (the model copies the first label regardless of content),
so a content-swap gives NO counterfactual (the old version: KL_cut 0.018, dead). We
instead flip the LABEL SCHEME, which the first-label-copier actually responds to:

    clean:     "Question: Which is the answer? Answer Choices: (A) {a} (B) {b} Answer: ("  -> "A"
    corrupted: "Question: Which is the answer? Answer Choices: (1) {a} (2) {b} Answer: ("  -> "1"
    metric: -(logit["A"] - logit["B"]) at the final position (the A-over-B preference).

Swapping A/B -> 1/2 changes the first label the model copies (A -> 1), so the output
flips and KL_cut should be healthy. RUN THE PROBE: KL_cut >~1.5 and faith@3000 clearing
~0.85 = viable; if KL_cut is still tiny the positional mechanism doesn't survive a label
relabel and the task is genuinely degenerate.
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
# clean labels (A,B) vs corrupted labels (1,2): the first label the model copies flips A->1.
_CLEAN = "Question: Which is the answer? Answer Choices: (A) {a} (B) {b} Answer: ("
_CORR = "Question: Which is the answer? Answer Choices: (1) {a} (2) {b} Answer: ("


def _single_tok(tok, s: str, lead: str = " ") -> bool:
    return len(tok(lead + s, add_special_tokens=False)["input_ids"]) == 1


def _tok_id(tok, s: str):
    ids = tok(s, add_special_tokens=False)["input_ids"]   # no leading space (follows "(")
    return ids[0] if len(ids) == 1 else None


def _build_mcq_batch(model, batch_size, seed, device):
    tok = model.tokenizer
    rng = random.Random(seed)
    opts = [w for w in _OPTIONS if _single_tok(tok, w)]
    if len(opts) < 4:
        raise RuntimeError("not enough single-token MCQ options")
    a_id, b_id = _tok_id(tok, "A"), _tok_id(tok, "B")     # clean answer "A", distractor "B"
    one_id = _tok_id(tok, "1")                            # corrupted first label "1" (sanity)
    if a_id is None or b_id is None or one_id is None:
        raise RuntimeError("'A'/'B'/'1' not single tokens")

    clean_ids, corrupt_ids, correct, wrong, target_len = [], [], [], [], None
    tries = 0
    while len(clean_ids) < batch_size and tries < batch_size * 200:
        tries += 1
        a, b = rng.sample(opts, 2)
        ci = tok(_CLEAN.format(a=a, b=b), add_special_tokens=False)["input_ids"]
        xi = tok(_CORR.format(a=a, b=b), add_special_tokens=False)["input_ids"]
        if len(ci) != len(xi):
            continue
        if target_len is None:
            target_len = len(ci)
        if len(ci) != target_len:
            continue
        clean_ids.append(ci); corrupt_ids.append(xi)
        correct.append(a_id); wrong.append(b_id)
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
    """First-label positional bias in multiple-choice (heads L8.H1, L10.H8 + MLP9)."""

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
