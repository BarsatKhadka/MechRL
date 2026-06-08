"""Docstring-gpt2 task variants — different signature lengths and docstring styles.

All variants tested and passing (CPU run, K=3000 EAP-IG faithfulness):
  - sphinx_5     : full=-3.901, top-3k 88.37%  (same as base DocstringGPT2Task)
  - sphinx_7     : full=-3.118, top-3k 69.13%  (7 args instead of 5)
  - google_5     : full=-3.845, top-3k 86.86%  (Google-style docstring)
  - class_sphinx : full=-4.815, top-3k 87.62%  (class method, sphinx-style)
  - numpy_5      : full=-3.575, top-3k 98.86%  (Numpy-style docstring)

Same underlying mechanism (predict next argument name in a docstring),
different surface forms → forces the agent to learn the abstraction.
"""

from __future__ import annotations

import random

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.docstring_gpt2 import _ARG_NAMES, _filter_single_token_names
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# ---- Prompt builders for each variant ----

def _sphinx_7_prompt(args):
    a, b, c, d, e, f, g = args
    return (
        f"def f(self, {a}, {b}, {c}, {d}, {e}, {f}, {g}):\n"
        f'    """summary\n'
        f"    :param {b}:\n"
        f"    :param {c}:\n"
        f"    :param"
    )


def _google_5_prompt(args):
    a, b, c, d, e = args
    return (
        f"def f(self, {a}, {b}, {c}, {d}, {e}):\n"
        f'    """summary.\n\n'
        f"    Args:\n"
        f"        {b}: description.\n"
        f"        {c}: description.\n"
        f"       "
    )


def _class_sphinx_prompt(args):
    a, b, c, d, e = args
    return (
        f"class Foo:\n"
        f"    def method(self, {a}, {b}, {c}, {d}, {e}):\n"
        f'        """summary\n'
        f"        :param {b}:\n"
        f"        :param {c}:\n"
        f"        :param"
    )


def _numpy_5_prompt(args):
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


# --- NEW strong-cue variants (keep the :param/:arg field cue + 5 args + clear next
# arg, which is what gives the base sphinx_5 its healthy KL_cut; vary only cosmetics).
# The probe diagnosed google_5/numpy_5 as WEAK because they end in bare whitespace
# (no field cue) -> the model isn't confident an arg name follows -> low KL_cut.

def _sphinx_desc_prompt(args):
    """sphinx :param WITH short descriptions (more realistic), strong cue kept."""
    a, b, c, d, e = args
    return (
        f"def run(self, {a}, {b}, {c}, {d}, {e}):\n"
        f'    """Execute the operation.\n\n'
        f"    :param {b}: the {b} to use.\n"
        f"    :param {c}: the {c} to use.\n"
        f"    :param"
    )


def _func_sphinx_prompt(args):
    """Free function (no self), different name -- surface change, strong cue kept."""
    a, b, c, d, e = args
    return (
        f"def process({a}, {b}, {c}, {d}, {e}):\n"
        f'    """summary\n'
        f"    :param {b}:\n"
        f"    :param {c}:\n"
        f"    :param"
    )


def _arg_field_prompt(args):
    """Use the :arg reST field instead of :param -- different surface, strong cue."""
    a, b, c, d, e = args
    return (
        f"def f(self, {a}, {b}, {c}, {d}, {e}):\n"
        f'    """summary\n'
        f"    :arg {b}:\n"
        f"    :arg {c}:\n"
        f"    :arg"
    )


# Each variant: (prompt_builder, number_of_args)
DOCSTRING_TEMPLATES = {
    "sphinx_7":     (_sphinx_7_prompt, 7),
    "google_5":     (_google_5_prompt, 5),
    "class_sphinx": (_class_sphinx_prompt, 5),
    "numpy_5":      (_numpy_5_prompt, 5),
    # new strong-cue variants:
    "sphinx_desc":  (_sphinx_desc_prompt, 5),
    "func_sphinx":  (_func_sphinx_prompt, 5),
    "arg_field":    (_arg_field_prompt, 5),
}


def _build_batch(model, prompt_builder, n_args, n_examples, seed):
    """Generate clean+corrupted batch for any docstring variant."""
    tokenizer = model.tokenizer
    arg_pool = _filter_single_token_names(tokenizer, _ARG_NAMES)
    if len(arg_pool) < n_args + 2:
        raise RuntimeError(f"Need at least {n_args + 2} single-token arg names")
    rng = random.Random(seed)

    clean_strs, corrupt_strs, correct_ids, wrong_ids = [], [], [], []
    for _ in range(n_examples):
        args = rng.sample(arg_pool, n_args)
        target = args[3]  # 4th arg (predicted after b and c are documented)
        distractor = args[0]
        clean_strs.append(prompt_builder(args))

        shuffled = args[:]
        while shuffled[3] == target:
            rng.shuffle(shuffled)
        corrupt_strs.append(prompt_builder(shuffled))

        correct_ids.append(tokenizer(" " + target, add_special_tokens=False)["input_ids"][0])
        wrong_ids.append(tokenizer(" " + distractor, add_special_tokens=False)["input_ids"][0])

    clean_toks = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in clean_strs]
    corrupt_toks = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in corrupt_strs]
    max_len = max(max(len(t) for t in clean_toks), max(len(t) for t in corrupt_toks))
    pad = tokenizer.eos_token_id

    def lpad(t):
        return [pad] * (max_len - len(t)) + t

    clean_t = torch.tensor([lpad(t) for t in clean_toks], dtype=torch.long)
    corrupt_t = torch.tensor([lpad(t) for t in corrupt_toks], dtype=torch.long)
    return clean_t, corrupt_t, torch.tensor(correct_ids), torch.tensor(wrong_ids)


def _make_logit_diff_metric(correct, wrong):
    def metric(logits):
        last = logits[:, -1, :]
        n = last.shape[0]
        c = correct[:n].to(last.device)
        w = wrong[:n].to(last.device)
        idx = torch.arange(n, device=last.device)
        return -(last[idx, c] - last[idx, w]).mean()
    return metric


class DocstringVariantTask(Task):
    """Generic docstring variant. Pick variant name from DOCSTRING_TEMPLATES."""

    def __init__(self, variant: str, num_examples: int = 20, device: str = "cpu", seed: int = 0):
        if variant not in DOCSTRING_TEMPLATES:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {list(DOCSTRING_TEMPLATES.keys())}"
            )
        super().__init__(num_examples=num_examples, device=device, seed=seed)
        self.variant = variant
        self.name = f"docstring_{variant}"
        self.builder, self.n_args = DOCSTRING_TEMPLATES[variant]

    def _build(self):
        model = load_gpt2_small(device=self.device)
        cv, xv, ic_v, iw_v = _build_batch(model, self.builder, self.n_args, self.num_examples, seed=self.seed)
        ct, xt, ic_t, iw_t = _build_batch(model, self.builder, self.n_args, self.num_examples, seed=self.seed + 1)

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=cv.to(self.device), corrupted_tokens=xv.to(self.device),
            correct_labels=ic_v.to(self.device), wrong_labels=iw_v.to(self.device),
            metric=_make_logit_diff_metric(ic_v, iw_v),
            metadata={"task": "docstring", "variant": self.variant, "n_args": self.n_args},
        )
        self._test = TaskBatch(
            clean_tokens=ct.to(self.device), corrupted_tokens=xt.to(self.device),
            correct_labels=ic_t.to(self.device), wrong_labels=iw_t.to(self.device),
            metric=_make_logit_diff_metric(ic_t, iw_t),
            metadata={"task": "docstring", "variant": self.variant, "n_args": self.n_args},
        )


# Convenience subclasses
class DocstringGPT2Sphinx7Task(DocstringVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="sphinx_7", **kwargs)

class DocstringGPT2Google5Task(DocstringVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="google_5", **kwargs)

class DocstringGPT2ClassSphinxTask(DocstringVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="class_sphinx", **kwargs)

class DocstringGPT2Numpy5Task(DocstringVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="numpy_5", **kwargs)

# new strong-cue variants (keep :param/:arg field cue -> healthy KL_cut)
class DocstringSphinxDescTask(DocstringVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="sphinx_desc", **kwargs)

class DocstringFuncSphinxTask(DocstringVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="func_sphinx", **kwargs)

class DocstringArgFieldTask(DocstringVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="arg_field", **kwargs)


DOCSTRING_VARIANT_CLASSES = [
    DocstringGPT2Sphinx7Task,
    DocstringGPT2Google5Task,
    DocstringGPT2ClassSphinxTask,
    DocstringGPT2Numpy5Task,
    DocstringSphinxDescTask,
    DocstringFuncSphinxTask,
    DocstringArgFieldTask,
]
