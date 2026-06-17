"""Country -> capital factual recall on GPT-2 small -- HELD-OUT candidate, ON PROBATION.

A clean factual-lookup behaviour GPT-2 small genuinely does ("The capital of France
is Paris"), and a DIFFERENT mechanism family (factual recall) from anything trained on.
No published SPARSE head/MLP circuit exists -- factual recall in GPT-2 is believed to
be MLP-heavy / diffuse (cf. ROME) -- so there is NO canonical-head recovery here; the
ceiling probe decides whether a sparse faithful subset exists at all. If it comes back
diffuse (faith only ~1.0 at K=ALL), we drop it. That is the honest test.

    clean:     "The capital of France is"  -> " Paris"
    corrupted: "The capital of Japan is"   -> " Tokyo"   (different country flips the answer)
    metric: -(logit[clean capital] - logit[corrupted capital]) at the final position.

Country + capital are both filtered to single tokens and the template is fixed, so
prompts are uniform length (rejection-built) -> healthy KL_cut from the country swap.
"""

from __future__ import annotations

import random
from typing import Callable

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# (country, capital) pairs; both filtered to single-token (leading space) at build.
_PAIRS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Russia", "Moscow"), ("Greece", "Athens"), ("Egypt", "Cairo"),
    ("China", "Beijing"), ("Cuba", "Havana"), ("Poland", "Warsaw"), ("Norway", "Oslo"),
    ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
]
_TEMPLATE = "The capital of {country} is"


def _single_tok(tok, s: str) -> bool:
    return len(tok(" " + s, add_special_tokens=False)["input_ids"]) == 1


def _build_fact_batch(model, batch_size, seed, device):
    tok = model.tokenizer
    rng = random.Random(seed)
    pairs = [(c, cap) for (c, cap) in _PAIRS if _single_tok(tok, c) and _single_tok(tok, cap)]
    if len(pairs) < 4:
        raise RuntimeError("not enough single-token country/capital pairs")
    cap_id = {cap: tok(" " + cap, add_special_tokens=False)["input_ids"][0] for _, cap in pairs}

    clean_ids, corrupt_ids, correct, wrong, target_len = [], [], [], [], None
    tries = 0
    while len(clean_ids) < batch_size and tries < batch_size * 200:
        tries += 1
        (c1, cap1), (c2, cap2) = rng.sample(pairs, 2)         # corrupted = different country
        ci = tok(_TEMPLATE.format(country=c1), add_special_tokens=False)["input_ids"]
        xi = tok(_TEMPLATE.format(country=c2), add_special_tokens=False)["input_ids"]
        if len(ci) != len(xi):
            continue
        if target_len is None:
            target_len = len(ci)
        if len(ci) != target_len:
            continue
        clean_ids.append(ci); corrupt_ids.append(xi)
        correct.append(cap_id[cap1]); wrong.append(cap_id[cap2])
    if len(clean_ids) < batch_size:
        raise RuntimeError(f"only built {len(clean_ids)}/{batch_size} country-capital prompts; add more pairs")
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


class CountryCapitalTask(Task):
    """Factual recall: country -> capital. On probation (likely diffuse; probe decides)."""

    name = "country_capital"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        cv, xv, icv, iwv = _build_fact_batch(model, self.num_examples, self.seed, self.device)
        ct, xt, ict, iwt = _build_fact_batch(model, self.num_examples, self.seed + 1, self.device)
        self._model = model
        meta = {"source": "factual recall (no published sparse circuit)", "task": self.name}
        self._validation = TaskBatch(
            clean_tokens=cv, corrupted_tokens=xv, correct_labels=icv, wrong_labels=iwv,
            metric=_metric(icv, iwv), metadata=meta)
        self._test = TaskBatch(
            clean_tokens=ct, corrupted_tokens=xt, correct_labels=ict, wrong_labels=iwt,
            metric=_metric(ict, iwt), metadata=meta)
