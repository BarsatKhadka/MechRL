"""Subject-Verb Agreement (verb conjugation) on GPT-2 small.

Circuit identified in arXiv:2506.22105 (12 heads / 7 layers, late-layer dominated)
and Finlayson et al. 2021. The SUBJECT's number sets the verb form; an "attractor"
noun of the opposite number sits in between (the classic agreement-attractor setup).

    clean (singular subj):  "The author near the cars"   -> predict " is"
    corrupted (plural subj): "The authors near the cars" -> predict " are"
    metric: -(logit[correct verb] - logit[wrong verb]) at the final position

The counterfactual flips the SUBJECT number (author<->authors), which flips is<->are
-> large output change -> healthy KL_cut. Sharp single-token answer. Every slot is
single-token so all prompts are the SAME length -> NO EOS padding (the artifact that
killed successor's KL_cut).

A different mechanism family (syntax) from IOI/greater-than/docstring -> a strong
cross-family transfer task.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

import torch
from transformer_lens import HookedTransformer

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# (singular, plural) subject pairs; filtered to BOTH single-token at build time.
_NOUN_PAIRS = [
    ("author", "authors"), ("key", "keys"), ("book", "books"), ("dog", "dogs"),
    ("car", "cars"), ("friend", "friends"), ("player", "players"), ("teacher", "teachers"),
    ("doctor", "doctors"), ("writer", "writers"), ("officer", "officers"), ("driver", "drivers"),
    ("farmer", "farmers"), ("painter", "painters"), ("singer", "singers"), ("worker", "workers"),
    ("student", "students"), ("manager", "managers"), ("owner", "owners"), ("leader", "leaders"),
]
_ATTRACTORS = ["cars", "books", "desks", "tables", "rooms", "doors", "trees", "walls", "roads", "boxes"]
_PREPS = ["near", "by", "behind", "beside"]
_VERB = {"sing": "is", "plural": "are"}


def _single_tok(tokenizer, s: str) -> bool:
    return len(tokenizer(" " + s, add_special_tokens=False)["input_ids"]) == 1


def _build_sva_batch(model, batch_size, seed, device):
    tok = model.tokenizer
    pairs = [(s, p) for (s, p) in _NOUN_PAIRS if _single_tok(tok, s) and _single_tok(tok, p)]
    attrs = [a for a in _ATTRACTORS if _single_tok(tok, a)]
    preps = [p for p in _PREPS if _single_tok(tok, p)]
    if len(pairs) < 5 or not attrs or not preps:
        raise RuntimeError("not enough single-token SVA slots")
    is_id = tok(" is", add_special_tokens=False)["input_ids"][0]
    are_id = tok(" are", add_special_tokens=False)["input_ids"][0]
    rng = random.Random(seed)

    clean_strs, corrupt_strs, correct_ids, wrong_ids = [], [], [], []
    for _ in range(batch_size):
        sing, plur = rng.choice(pairs)
        attr, prep = rng.choice(attrs), rng.choice(preps)
        singular = rng.random() < 0.5
        subj, subj_cf = (sing, plur) if singular else (plur, sing)   # corrupted flips number
        clean_strs.append(f"The {subj} {prep} the {attr}")
        corrupt_strs.append(f"The {subj_cf} {prep} the {attr}")
        correct_ids.append(is_id if singular else are_id)
        wrong_ids.append(are_id if singular else is_id)

    clean = [tok(s, add_special_tokens=False)["input_ids"] for s in clean_strs]
    corrupt = [tok(s, add_special_tokens=False)["input_ids"] for s in corrupt_strs]
    L = clean[0]
    assert all(len(c) == len(L) and len(x) == len(L) for c, x in zip(clean, corrupt)), \
        "SVA prompts not uniform length -- a slot tokenized to >1 token"
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


class SubjectVerbAgreementTask(Task):
    """Subject-verb number agreement (is/are) on GPT-2 small."""

    name = "subject_verb"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        cv, xv, ic_v, iw_v = _build_sva_batch(model, self.num_examples, self.seed, self.device)
        ct, xt, ic_t, iw_t = _build_sva_batch(model, self.num_examples, self.seed + 1, self.device)
        self._model = model
        self._validation = TaskBatch(
            clean_tokens=cv, corrupted_tokens=xv, correct_labels=ic_v, wrong_labels=iw_v,
            metric=_metric(ic_v, iw_v),
            metadata={"source": "arXiv:2506.22105 / Finlayson 2021", "task": "subject-verb agreement"},
        )
        self._test = TaskBatch(
            clean_tokens=ct, corrupted_tokens=xt, correct_labels=ic_t, wrong_labels=iw_t,
            metric=_metric(ic_t, iw_t),
            metadata={"source": "arXiv:2506.22105 / Finlayson 2021", "task": "subject-verb agreement"},
        )
