"""Gendered Pronoun prediction on GPT-2 small (Mathwin et al., MI hackathon; ACDC).

    clean (male name):    "So Dave is a really great friend, isn't"  -> predict " he"
    corrupted (female):   "So Sarah is a really great friend, isn't" -> predict " she"
    metric: -(logit[correct pronoun] - logit[wrong pronoun]) at the final position

The counterfactual swaps the name's gender -> flips he<->she -> healthy KL_cut.
All names single-token + fixed template -> uniform length, NO EOS padding.

NOTE: this circuit is MLP-heavy (Mathwin: MLPs matter more than heads), so the
top-K ceiling MAY be lower than IOI -- the probe decides if it clears the gate.
"""

from __future__ import annotations

import random
from typing import Callable

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


_MALE = ["John", "Mike", "Dave", "Paul", "Mark", "Tom", "Steve", "Brian", "Kevin",
         "Jason", "Gary", "Eric", "Adam", "Scott", "Frank", "Peter", "Greg", "Carl"]
_FEMALE = ["Mary", "Sarah", "Anna", "Laura", "Emma", "Julia", "Linda", "Karen", "Susan",
           "Alice", "Diana", "Nancy", "Carol", "Helen", "Janet", "Donna", "Lisa", "Amy"]
_TEMPLATE = "So {name} is a really great friend, isn't"   # -> " he" / " she"


def _single_tok(tokenizer, s: str) -> bool:
    return len(tokenizer(" " + s, add_special_tokens=False)["input_ids"]) == 1


def _build_pronoun_batch(model, batch_size, seed, device):
    tok = model.tokenizer
    male = [n for n in _MALE if _single_tok(tok, n)]
    female = [n for n in _FEMALE if _single_tok(tok, n)]
    if len(male) < 5 or len(female) < 5:
        raise RuntimeError("not enough single-token names")
    he_id = tok(" he", add_special_tokens=False)["input_ids"][0]
    she_id = tok(" she", add_special_tokens=False)["input_ids"][0]
    rng = random.Random(seed)

    clean_strs, corrupt_strs, correct_ids, wrong_ids = [], [], [], []
    for _ in range(batch_size):
        male_subj = rng.random() < 0.5
        if male_subj:
            name, name_cf = rng.choice(male), rng.choice(female)   # corrupted flips gender
        else:
            name, name_cf = rng.choice(female), rng.choice(male)
        clean_strs.append(_TEMPLATE.format(name=name))
        corrupt_strs.append(_TEMPLATE.format(name=name_cf))
        correct_ids.append(he_id if male_subj else she_id)
        wrong_ids.append(she_id if male_subj else he_id)

    clean = [tok(s, add_special_tokens=False)["input_ids"] for s in clean_strs]
    corrupt = [tok(s, add_special_tokens=False)["input_ids"] for s in corrupt_strs]
    L = len(clean[0])
    assert all(len(c) == L and len(x) == L for c, x in zip(clean, corrupt)), \
        "pronoun prompts not uniform length -- a name tokenized to >1 token"
    return (torch.tensor(clean, dtype=torch.long, device=device),
            torch.tensor(corrupt, dtype=torch.long, device=device),
            torch.tensor(correct_ids, dtype=torch.long, device=device),
            torch.tensor(wrong_ids, dtype=torch.long, device=device))


def _metric(correct: torch.Tensor, wrong: torch.Tensor) -> Callable:
    def m(logits):
        last = logits[:, -1, :]
        n = last.shape[0]
        c, w = correct[:n].to(last.device), wrong[:n].to(last.device)
        idx = torch.arange(n, device=last.device)
        return -(last[idx, c] - last[idx, w]).mean()
    return m


class GenderedPronounTask(Task):
    """Predict he/she from a single-token name's gender."""

    name = "gendered_pronoun"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        cv, xv, ic_v, iw_v = _build_pronoun_batch(model, self.num_examples, self.seed, self.device)
        ct, xt, ic_t, iw_t = _build_pronoun_batch(model, self.num_examples, self.seed + 1, self.device)
        self._model = model
        self._validation = TaskBatch(
            clean_tokens=cv, corrupted_tokens=xv, correct_labels=ic_v, wrong_labels=iw_v,
            metric=_metric(ic_v, iw_v),
            metadata={"source": "Mathwin et al. (MI hackathon) / ACDC", "task": "gendered pronoun"},
        )
        self._test = TaskBatch(
            clean_tokens=ct, corrupted_tokens=xt, correct_labels=ic_t, wrong_labels=iw_t,
            metric=_metric(ic_t, iw_t),
            metadata={"source": "Mathwin et al. (MI hackathon) / ACDC", "task": "gendered pronoun"},
        )
