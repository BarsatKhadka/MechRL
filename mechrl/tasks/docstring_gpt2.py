"""Docstring task adapted to run on GPT-2 small.

Heimersheim & Janiak's published "docstring circuit" exists in attn-only-4l
(a tiny model they trained specifically on docstring-like patterns). To use
docstring as a *training* task on GPT-2 small, we adapt the prompt style and
score predictions GPT-2 makes natively.

Prompt template:
    def f(self, {a}, {b}, {c}, {d}, {e}):
        \"\"\"docstring summary
        :param {b}: <description>
        :param {c}: <description>
        :param

The model should predict ` {d}` — the next argument name in order.

Clean: real argument-name sequence (a, b, c, d, e).
Corrupted: shuffled argument names so the model has no order signal.
Metric: -logit_diff between correct ({d}) and a wrong-name distractor.

NOTE: there is no published "ground truth circuit" for this task on GPT-2
small — Heimersheim only analyzed attn-only-4l. We use this for training
diversity. Don't include it in Stage 2 reward validation (which requires
canonical published circuits).
"""

from __future__ import annotations

import random
from typing import Callable, List

import torch
from transformer_lens import HookedTransformer

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# Common variable/argument names that tokenize as single GPT-2 tokens when
# preceded by a space (so the metric can index them as one-token answers).
_ARG_NAMES = [
    "data", "file", "name", "value", "key", "result", "state",
    "size", "shape", "index", "config", "model", "input", "output",
    "label", "args", "path", "node", "user", "type", "mode", "table",
    "count", "score", "level", "host", "port", "token", "image", "text",
    "field", "query", "limit", "buffer", "stream", "session", "client",
    "context", "request", "header", "filter", "format", "version",
]


def _filter_single_token_names(tokenizer, candidates: List[str]) -> List[str]:
    """Keep only names that tokenize to a single GPT-2 token when prefixed with ' '."""
    out = []
    for name in candidates:
        toks = tokenizer(" " + name, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            out.append(name)
    return out


def _build_docstring_prompt(args: List[str]) -> str:
    """Return prompt text up to and including the `:param ` that should be
    completed with the next argument name. Predictable structure for logit-diff.
    """
    assert len(args) == 5
    a, b, c, d, _e = args
    return (
        f"def f(self, {a}, {b}, {c}, {d}):\n"
        f'    """summary\n'
        f"    :param {b}:\n"
        f"    :param {c}:\n"
        f"    :param"
    )


def _build_docstring_batch(
    model: HookedTransformer,
    batch_size: int,
    arg_pool: List[str],
    seed: int,
    device: str,
):
    """Generate clean (real order) and corrupted (shuffled) prompts.

    correct_label = id of `" d"` token (the next arg name in original order)
    wrong_label   = id of `" a"` token (an earlier param, plausible distractor)
    """
    tokenizer = model.tokenizer
    rng = random.Random(seed)

    clean_strs, corrupt_strs, correct_ids, wrong_ids = [], [], [], []
    for _ in range(batch_size):
        # Sample 5 distinct argument names
        args = rng.sample(arg_pool, 5)
        a, b, c, d, e = args

        # Clean: real signature
        clean_strs.append(_build_docstring_prompt(args))

        # Corrupted: same names but shuffled in the signature, so the "next param"
        # cue (b, c documented → d should come next) breaks.
        shuffled = args[:]
        rng.shuffle(shuffled)
        # Ensure the corrupted ordering actually differs at position 3 (d's slot)
        while shuffled[3] == d:
            rng.shuffle(shuffled)
        corrupt_strs.append(_build_docstring_prompt(shuffled))

        correct_id = tokenizer(" " + d, add_special_tokens=False)["input_ids"][0]
        wrong_id = tokenizer(" " + a, add_special_tokens=False)["input_ids"][0]
        correct_ids.append(correct_id)
        wrong_ids.append(wrong_id)

    # Tokenize prompts. They should all be the same length because the template
    # is fixed and arg names are single tokens.
    clean_tok_lists = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in clean_strs]
    corrupt_tok_lists = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in corrupt_strs]
    target_len = max(len(t) for t in clean_tok_lists)

    # Pad to target_len with the EOS token (we measure at the LAST non-pad position,
    # but since template is fixed, all sequences should already be target_len).
    def pad(t):
        return t + [tokenizer.eos_token_id] * (target_len - len(t))

    clean_tok_lists = [pad(t) for t in clean_tok_lists]
    corrupt_tok_lists = [pad(t) for t in corrupt_tok_lists]

    clean_tokens = torch.tensor(clean_tok_lists, dtype=torch.long, device=device)
    corrupt_tokens = torch.tensor(corrupt_tok_lists, dtype=torch.long, device=device)
    correct_t = torch.tensor(correct_ids, dtype=torch.long, device=device)
    wrong_t = torch.tensor(wrong_ids, dtype=torch.long, device=device)
    return clean_tokens, corrupt_tokens, correct_t, wrong_t


def _build_logit_diff_metric(correct: torch.Tensor, wrong: torch.Tensor) -> Callable:
    """Metric: -(logit[correct] - logit[wrong]) at final position, averaged."""

    def metric(logits: torch.Tensor) -> torch.Tensor:
        last_logits = logits[:, -1, :]
        n = last_logits.shape[0]
        c = correct[:n].to(last_logits.device)
        w = wrong[:n].to(last_logits.device)
        idx = torch.arange(n, device=last_logits.device)
        return -(last_logits[idx, c] - last_logits[idx, w]).mean()

    return metric


class DocstringGPT2Task(Task):
    """Docstring-style argument prediction on GPT-2 small."""

    name = "docstring_gpt2"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        arg_pool = _filter_single_token_names(model.tokenizer, _ARG_NAMES)
        if len(arg_pool) < 10:
            raise RuntimeError(
                f"Only {len(arg_pool)} single-token arg names — need more for variety."
            )

        clean_v, corrupt_v, correct_v, wrong_v = _build_docstring_batch(
            model, self.num_examples, arg_pool, seed=self.seed, device=self.device
        )
        clean_t, corrupt_t, correct_t, wrong_t = _build_docstring_batch(
            model, self.num_examples, arg_pool, seed=self.seed + 1, device=self.device
        )

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=clean_v,
            corrupted_tokens=corrupt_v,
            correct_labels=correct_v,
            wrong_labels=wrong_v,
            metric=_build_logit_diff_metric(correct_v, wrong_v),
            metadata={
                "source": "adapted from Heimersheim & Janiak (2023) for GPT-2",
                "task": "predict next docstring argument name",
                "arg_pool_size": len(arg_pool),
            },
        )
        self._test = TaskBatch(
            clean_tokens=clean_t,
            corrupted_tokens=corrupt_t,
            correct_labels=correct_t,
            wrong_labels=wrong_t,
            metric=_build_logit_diff_metric(correct_t, wrong_t),
            metadata={
                "source": "adapted from Heimersheim & Janiak (2023) for GPT-2",
                "task": "predict next docstring argument name",
                "arg_pool_size": len(arg_pool),
            },
        )
