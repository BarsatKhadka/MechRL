"""Binary syllogism truth-value prediction on GPT-2 small (Saraipour & Zhang 2025,
arXiv:2508.16109) -- two HELD-OUT tasks.

GPT-2 small completes simple two-statement syllogisms with a truth token. The
circuit is a small set of late-layer "truth heads"; a different mechanism family
(logical / truth-value) from anything in the training pool, so a strong
cross-mechanism transfer test.

  SIMPLE  (B matches A -> B has A's truth value)
    clean:     "Statement A is true. Statement B matches statement A. Statement B is"  -> " true"
    corrupted: "Statement A is false. Statement B matches statement A. Statement B is" -> " false"
    Truth heads (paper Fig 2b): 7.2, 9.1, 9.9, 10.1, 10.4 (~3-5 recover >90%).

  OPPOSITE  (B opposite of A -> B has A's NEGATED truth value)
    clean:     "Statement A and statement B are opposite. Statement A is true. Statement B is"  -> " false"
    corrupted: "Statement A and statement B are opposite. Statement A is false. Statement B is" -> " true"
    Negative-truth heads: 7.3, 8.8, 8.10, 9.7, 10.7  +  rescaler MLPs 7,8,9,10 (~85% faith).

  metric: -(logit[correct truth] - logit[wrong truth]) at the final position.

The counterfactual flips the stated truth value, which flips the answer -> healthy
KL_cut. " true" and " false" are both single tokens and the template is otherwise
fixed, so all prompts are the same length -> NO EOS padding.

CAVEAT (paper): the OPPOSITE/negation mechanism fires reliably for true->false but
less so for false->true. RUN A CEILING PROBE (scripts.probe_task_ceiling) before
trusting either: healthy KL_cut + top-3000 faith clearing ~0.85 = viable held-out task.
Source is a 2025 preprint -- cite honestly.
"""

from __future__ import annotations

import random
from typing import Callable

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


def _truth_ids(tok):
    t = tok(" true", add_special_tokens=False)["input_ids"]
    f = tok(" false", add_special_tokens=False)["input_ids"]
    if len(t) != 1 or len(f) != 1:
        raise RuntimeError(f"' true'/' false' not single-token (got {t}, {f})")
    return t[0], f[0]


def _build_batch(model, batch_size, seed, device, opposite: bool):
    tok = model.tokenizer
    true_id, false_id = _truth_ids(tok)
    rng = random.Random(seed)

    clean_strs, corrupt_strs, correct_ids, wrong_ids = [], [], [], []
    for _ in range(batch_size):
        a_true = rng.random() < 0.5
        a_word = "true" if a_true else "false"
        a_word_cf = "false" if a_true else "true"          # corrupted flips A's truth
        if opposite:
            # B is the negation of A; the answer is the opposite of A's stated truth.
            ans_id = false_id if a_true else true_id
            wrong = true_id if a_true else false_id
            clean_strs.append(
                f"Statement A and statement B are opposite. Statement A is {a_word}. Statement B is")
            corrupt_strs.append(
                f"Statement A and statement B are opposite. Statement A is {a_word_cf}. Statement B is")
        else:
            # B matches A; the answer is A's stated truth.
            ans_id = true_id if a_true else false_id
            wrong = false_id if a_true else true_id
            clean_strs.append(
                f"Statement A is {a_word}. Statement B matches statement A. Statement B is")
            corrupt_strs.append(
                f"Statement A is {a_word_cf}. Statement B matches statement A. Statement B is")
        correct_ids.append(ans_id)
        wrong_ids.append(wrong)

    clean = [tok(s, add_special_tokens=False)["input_ids"] for s in clean_strs]
    corrupt = [tok(s, add_special_tokens=False)["input_ids"] for s in corrupt_strs]
    L = len(clean[0])
    assert all(len(c) == L and len(x) == L for c, x in zip(clean, corrupt)), \
        "syllogism prompts not uniform length -- a truth word tokenized to >1 token"
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


class _SyllogismBase(Task):
    _opposite: bool = False
    _source: str = "Saraipour & Zhang 2025 (arXiv:2508.16109)"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        cv, xv, icv, iwv = _build_batch(model, self.num_examples, self.seed, self.device, self._opposite)
        ct, xt, ict, iwt = _build_batch(model, self.num_examples, self.seed + 1, self.device, self._opposite)
        self._model = model
        meta = {"source": self._source, "task": self.name}
        self._validation = TaskBatch(
            clean_tokens=cv, corrupted_tokens=xv, correct_labels=icv, wrong_labels=iwv,
            metric=_metric(icv, iwv), metadata=meta)
        self._test = TaskBatch(
            clean_tokens=ct, corrupted_tokens=xt, correct_labels=ict, wrong_labels=iwt,
            metric=_metric(ict, iwt), metadata=meta)


class SimpleSyllogismTask(_SyllogismBase):
    """B matches A -> predict A's truth value (truth heads)."""
    name = "simple_syllogism"
    _opposite = False


class OppositeSyllogismTask(_SyllogismBase):
    """B opposite of A -> predict A's negated truth value (negative-truth heads + MLPs)."""
    name = "opposite_syllogism"
    _opposite = True
