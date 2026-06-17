"""Acronym prediction on GPT-2 small (Garc\'ia-Carrasco et al. 2024) -- HELD-OUT task.

The model reads a capitalised multi-word phrase and must predict the next letter
of its acronym, using "letter-mover" heads that copy each word's initial.

    clean:     "The Cardiac Arrest Symptoms (CA"  -> predict "S"  (Symptoms)
    corrupted: "The Cardiac Arrest Disease  (CA"  -> predict "D"  (Disease)
    metric: -(logit[correct letter] - logit[wrong letter]) at the final position

The counterfactual swaps the LAST word for one with a different initial, flipping
the answer letter while keeping the first two letters fixed -> healthy KL_cut.

Held-out / transfer task. Known circuit: Garc\'ia-Carrasco, Acerbo, et al. (2024),
"How does GPT-2 Predict Acronyms?" -- a small set of letter-mover attention heads
in GPT-2 small. SKETCH: tokenisation of the letter prefix is fiddly, so we build by
rejection, keeping only triples whose clean/corrupted prompts share a token length
and whose answer letters are single tokens. RUN A CEILING PROBE before trusting it:
(1) check the full model actually does it (validation logit-diff > 0), then
(2) check top-3000 KL-faith clears ~0.85, the same gate as the training tasks.
"""

from __future__ import annotations

import random
from typing import Callable

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# Capitalised words; the builder keeps only those that are a single token with a
# leading space, then groups them by initial. Extend freely -- more words, more
# distinct initials, easier to build uniform-length batches.
_WORDS = [
    "Cardiac", "Arrest", "Symptoms", "Disease", "National", "Science", "Foundation",
    "Central", "Banking", "System", "Global", "Energy", "Market", "Public", "Health",
    "Service", "Federal", "Reserve", "Board", "Digital", "Media", "Network", "Modern",
    "Museum", "World", "Trade", "Center", "Human", "Rights", "Open", "Source", "Data",
    "Analysis", "Group", "Future", "Power", "Water", "Forest", "Mountain", "River",
    "Animal", "Control", "Office", "Justice", "Defense", "Labor", "Education", "Housing",
]


def _letter_id(tok, letter: str):
    ids = tok(letter, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def _build_acronym_batch(model, batch_size, seed, device):
    tok = model.tokenizer
    rng = random.Random(seed)

    words = [w for w in _WORDS
             if len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1]
    by_init = {}
    for w in words:
        by_init.setdefault(w[0], []).append(w)
    inits = [k for k, v in by_init.items() if _letter_id(tok, k) is not None]
    if len(inits) < 4:
        raise RuntimeError("not enough single-token words / initials for acronyms")

    clean_ids, corrupt_ids, correct, wrong = [], [], [], []
    target_len, tries = None, 0
    while len(clean_ids) < batch_size and tries < batch_size * 400:
        tries += 1
        l1, l2, l3 = rng.sample(inits, 3)                          # distinct initials
        l3p = rng.choice([x for x in inits if x != l3])            # swapped initial
        w1, w2, w3 = (rng.choice(by_init[l1]), rng.choice(by_init[l2]),
                      rng.choice(by_init[l3]))
        w3p = rng.choice(by_init[l3p])
        clean_s = f"The {w1} {w2} {w3} ({l1}{l2}"
        corr_s = f"The {w1} {w2} {w3p} ({l1}{l2}"
        ci = tok(clean_s, add_special_tokens=False)["input_ids"]
        xi = tok(corr_s, add_special_tokens=False)["input_ids"]
        if len(ci) != len(xi):
            continue                                               # need matched length
        if target_len is None:
            target_len = len(ci)
        if len(ci) != target_len:
            continue                                               # uniform across batch
        c_id, w_id = _letter_id(tok, l3), _letter_id(tok, l3p)
        if c_id is None or w_id is None:
            continue
        clean_ids.append(ci); corrupt_ids.append(xi)
        correct.append(c_id); wrong.append(w_id)

    if len(clean_ids) < batch_size:
        raise RuntimeError(f"only built {len(clean_ids)}/{batch_size} acronym prompts; add more words")
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


class AcronymTask(Task):
    """Predict the next acronym letter (letter-mover circuit)."""

    name = "acronyms"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        cv, xv, icv, iwv = _build_acronym_batch(model, self.num_examples, self.seed, self.device)
        ct, xt, ict, iwt = _build_acronym_batch(model, self.num_examples, self.seed + 1, self.device)
        self._model = model
        self._validation = TaskBatch(
            clean_tokens=cv, corrupted_tokens=xv, correct_labels=icv, wrong_labels=iwv,
            metric=_metric(icv, iwv),
            metadata={"source": "Garcia-Carrasco et al. 2024", "task": "acronyms"},
        )
        self._test = TaskBatch(
            clean_tokens=ct, corrupted_tokens=xt, correct_labels=ict, wrong_labels=iwt,
            metric=_metric(ict, iwt),
            metadata={"source": "Garcia-Carrasco et al. 2024", "task": "acronyms"},
        )
