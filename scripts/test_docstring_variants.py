"""Gate 1 + Gate 2 for docstring-gpt2 variants.

5 candidate variants:
  1. original_sphinx   — 5 args, sphinx style (:param x:), predict 4th
  2. seven_args        — 7 args, sphinx style, predict 4th
  3. google_style      — 5 args, Google style (Args: x: ...), predict 4th
  4. class_method      — inside class Foo, 5 args, sphinx style
  5. numpy_style       — 5 args, Numpy style (Parameters / --------)

Each ends right after ":param" or equivalent, so model predicts the next arg.
"""

from __future__ import annotations

import random
from typing import Callable, List

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small
from mechrl.tasks.docstring_gpt2 import _ARG_NAMES, _filter_single_token_names
from mechrl.env import build_graph, Prefilter, AblationEngine


# Each builder returns the prompt string given the arg list. They differ in
# format but all end at the position where the next arg name should be predicted.
def _sphinx_5(args):  # original
    a, b, c, d, e = args
    return (
        f"def f(self, {a}, {b}, {c}, {d}, {e}):\n"
        f'    """summary\n'
        f"    :param {b}:\n"
        f"    :param {c}:\n"
        f"    :param"
    )


def _sphinx_7(args):
    a, b, c, d, e, f, g = args
    return (
        f"def f(self, {a}, {b}, {c}, {d}, {e}, {f}, {g}):\n"
        f'    """summary\n'
        f"    :param {b}:\n"
        f"    :param {c}:\n"
        f"    :param"
    )


def _google_5(args):
    a, b, c, d, e = args
    return (
        f"def f(self, {a}, {b}, {c}, {d}, {e}):\n"
        f'    """summary.\n\n'
        f"    Args:\n"
        f"        {b}: description.\n"
        f"        {c}: description.\n"
        f"       "
    )


def _class_sphinx_5(args):
    a, b, c, d, e = args
    return (
        f"class Foo:\n"
        f"    def method(self, {a}, {b}, {c}, {d}, {e}):\n"
        f'        """summary\n'
        f"        :param {b}:\n"
        f"        :param {c}:\n"
        f"        :param"
    )


def _numpy_5(args):
    a, b, c, d, e = args
    return (
        f"def f(self, {a}, {b}, {c}, {d}, {e}):\n"
        f'    """summary.\n\n'
        f"    Parameters\n"
        f"    ----------\n"
        f"    {b} : type\n"
        f"        desc.\n"
        f"    {c} : type\n"
        f"        desc.\n"
        f"   "
    )


VARIANTS = {
    "sphinx_5":     (_sphinx_5, 5),
    "sphinx_7":     (_sphinx_7, 7),
    "google_5":     (_google_5, 5),
    "class_sphinx": (_class_sphinx_5, 5),
    "numpy_5":      (_numpy_5, 5),
}


def build_batch(model, prompt_builder, n_args, n_examples, seed):
    tokenizer = model.tokenizer
    arg_pool = _filter_single_token_names(tokenizer, _ARG_NAMES)
    if len(arg_pool) < n_args + 2:
        raise RuntimeError("Need more single-token arg names")
    rng = random.Random(seed)

    clean_strs, corrupt_strs, correct_ids, wrong_ids = [], [], [], []
    for _ in range(n_examples):
        args = rng.sample(arg_pool, n_args)
        # Target is the 4th arg (index 3 = "d") in sphinx/google/numpy
        # because we document b and c, then predict d
        target = args[3]
        a = args[0]  # plausible distractor (earlier param)

        clean_strs.append(prompt_builder(args))

        # Corrupted: shuffled args
        shuffled = args[:]
        while shuffled[3] == target:
            rng.shuffle(shuffled)
        corrupt_strs.append(prompt_builder(shuffled))

        correct_ids.append(tokenizer(" " + target, add_special_tokens=False)["input_ids"][0])
        wrong_ids.append(tokenizer(" " + a, add_special_tokens=False)["input_ids"][0])

    # Tokenize, pad to same length
    clean_toks = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in clean_strs]
    corrupt_toks = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in corrupt_strs]
    max_len = max(max(len(t) for t in clean_toks), max(len(t) for t in corrupt_toks))
    pad = tokenizer.eos_token_id

    def lpad(t):
        return [pad] * (max_len - len(t)) + t

    clean_t = torch.tensor([lpad(t) for t in clean_toks], dtype=torch.long)
    corrupt_t = torch.tensor([lpad(t) for t in corrupt_toks], dtype=torch.long)
    return clean_t, corrupt_t, torch.tensor(correct_ids), torch.tensor(wrong_ids)


def make_metric(correct, wrong):
    def metric(logits):
        last = logits[:, -1, :]
        n = last.shape[0]
        c = correct[:n].to(last.device)
        w = wrong[:n].to(last.device)
        idx = torch.arange(n, device=last.device)
        return -(last[idx, c] - last[idx, w]).mean()
    return metric


class _DocstringVariantTask(Task):
    def __init__(self, variant_name, builder, n_args, num_examples=20, device="cpu", seed=0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)
        self.name = f"docstring_{variant_name}"
        self.builder = builder
        self.n_args = n_args

    def _build(self):
        model = load_gpt2_small(device=self.device)
        cv, xv, ic_v, iw_v = build_batch(model, self.builder, self.n_args, self.num_examples, seed=self.seed)
        ct, xt, ic_t, iw_t = build_batch(model, self.builder, self.n_args, self.num_examples, seed=self.seed + 1)

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=cv.to(self.device), corrupted_tokens=xv.to(self.device),
            correct_labels=ic_v.to(self.device), wrong_labels=iw_v.to(self.device),
            metric=make_metric(ic_v, iw_v),
            metadata={"variant": self.name},
        )
        self._test = TaskBatch(
            clean_tokens=ct.to(self.device), corrupted_tokens=xt.to(self.device),
            correct_labels=ic_t.to(self.device), wrong_labels=iw_t.to(self.device),
            metric=make_metric(ic_t, iw_t),
            metadata={"variant": self.name},
        )


def main():
    print(f"{'variant':>20} | {'seq_len':>7} | {'full':>7} | {'cut':>7} | {'gate1 logit-diff':>16} | {'gate2 top-3k':>14}")
    print(f"{'-'*20} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*16} | {'-'*14}")
    for vname, (builder, n_args) in VARIANTS.items():
        try:
            task = _DocstringVariantTask(vname, builder, n_args, num_examples=20, device="cpu")
            graph = build_graph(task.model)
            engine = AblationEngine(task, graph)
            full = engine.full_baseline()
            cut = engine.corrupted_baseline()
            logit_diff = -full
            gate1_pass = logit_diff > 2.0
            seq_len = task.validation_batch().seq_len
            if not gate1_pass:
                print(f"{vname:>20} | {seq_len:>7} | {full:>+7.3f} | {cut:>+7.3f} | {logit_diff:>9.3f} [FAIL] | (skip)")
                continue
            pref = Prefilter(task, graph, ig_steps=5)
            pref.compute(batch_size=10)
            f = engine.faithfulness(pref.candidate_mask(3000))
            g1_str = f"{logit_diff:>7.3f} [PASS]"
            g2_str = f"{f:>7.2%} [{'PASS' if f > 0.6 else 'FAIL'}]"
            print(f"{vname:>20} | {seq_len:>7} | {full:>+7.3f} | {cut:>+7.3f} | {g1_str:>16} | {g2_str:>14}")
        except Exception as e:
            msg = str(e).encode("ascii", errors="replace").decode("ascii")[:40]
            print(f"{vname:>20} | ERROR: {msg}")


if __name__ == "__main__":
    main()
