"""Docstring task from Heimersheim & Janiak (2023).

Wraps ACDC's `get_all_docstring_things`. The model is `attn-only-4l` (a small
4-layer attention-only transformer, NOT GPT-2 small).

Prompts look like:
    def f(self, files, obj, state, size, shape, option):
        \"\"\"document string example
        :param state: performance analysis
        :param size: pattern design
        :param

The model should predict the next argument name (e.g. " shape") at the final
position, based on having seen the function signature.

NOTE: Because this task runs on a different model than IOI/greater-than/
induction, the computational graph (number of nodes/edges) is different. Mixing
this task into the env requires the env to handle per-task models — additional
complexity. Provided here for completeness; consider whether to include it in
Stage 3 training.
"""

from __future__ import annotations

from acdc.docstring.utils import get_all_docstring_things

from mechrl.tasks.base import Task, TaskBatch


class DocstringTask(Task):
    name = "docstring"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        things = get_all_docstring_things(
            num_examples=self.num_examples,
            seq_len=41,
            device=self.device,
            metric_name="kl_div",
            correct_incorrect_wandb=False,
        )

        self._model = things.tl_model

        self._validation = TaskBatch(
            clean_tokens=things.validation_data,
            corrupted_tokens=things.validation_patch_data,
            correct_labels=things.validation_labels,
            wrong_labels=None,
            metric=things.validation_metric,
            metadata={
                "source": "acdc.docstring.utils.get_all_docstring_things",
                "model": "attn-only-4l (NOT GPT-2 small)",
            },
        )
        self._test = TaskBatch(
            clean_tokens=things.test_data,
            corrupted_tokens=things.test_patch_data,
            correct_labels=things.test_labels,
            wrong_labels=None,
            metric=things.test_metrics["kl_div"],
            metadata={
                "source": "acdc.docstring.utils.get_all_docstring_things",
                "model": "attn-only-4l (NOT GPT-2 small)",
            },
        )
