"""Tracr-compiled tasks from Lindner et al. 2023 (DeepMind).

Tracr compiles RASP programs into transformer weights, giving us tiny
transformers with mathematically exact ground-truth circuits.

Two tasks:
  - reverse: input list [a, b, c] -> output [c, b, a]
  - proportion: input ['a', 'x', 'b', 'x'] -> output [0, 0.5, 0.33, 0.5]

Both run on tiny tracr-compiled transformers (NOT GPT-2 small). They use
L2/MSE metrics rather than logit-diff. Including them gives us more diversity
in task structure even if they're on different models.

WARNING: reverse task is fixed at exactly 6 examples (all permutations of
[0, 1, 2]). proportion supports more examples.
"""

from __future__ import annotations

from acdc.tracr_task.utils import get_all_tracr_things

from mechrl.tasks.base import Task, TaskBatch


class TracrReverseTask(Task):
    name = "tracr_reverse"

    def __init__(self, num_examples: int = 6, device: str = "cpu", seed: int = 0):
        # The reverse task only supports num_examples=6 (all permutations)
        super().__init__(num_examples=6, device=device, seed=seed)

    def _build(self) -> None:
        things = get_all_tracr_things(
            task="reverse",
            metric_name="l2",
            num_examples=self.num_examples,
            device=self.device,
        )
        self._model = things.tl_model

        self._validation = TaskBatch(
            clean_tokens=things.validation_data,
            corrupted_tokens=things.validation_patch_data,
            correct_labels=things.validation_labels,
            wrong_labels=None,
            metric=things.validation_metric,
            metadata={
                "source": "acdc.tracr_task.utils.get_all_tracr_things(reverse)",
                "model": "tracr-compiled tiny transformer (NOT GPT-2 small)",
                "task_description": "reverse [a, b, c] -> [c, b, a]",
            },
        )
        self._test = TaskBatch(
            clean_tokens=things.test_data,
            corrupted_tokens=things.test_patch_data,
            correct_labels=things.test_labels,
            wrong_labels=None,
            metric=things.test_metrics["l2"],
            metadata={
                "source": "acdc.tracr_task.utils.get_all_tracr_things(reverse)",
                "model": "tracr-compiled tiny transformer",
            },
        )


class TracrProportionTask(Task):
    name = "tracr_proportion"

    def __init__(self, num_examples: int = 50, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        things = get_all_tracr_things(
            task="proportion",
            metric_name="l2",
            num_examples=self.num_examples,
            device=self.device,
        )
        self._model = things.tl_model

        self._validation = TaskBatch(
            clean_tokens=things.validation_data,
            corrupted_tokens=things.validation_patch_data,
            correct_labels=things.validation_labels,
            wrong_labels=None,
            metric=things.validation_metric,
            metadata={
                "source": "acdc.tracr_task.utils.get_all_tracr_things(proportion)",
                "model": "tracr-compiled tiny transformer (NOT GPT-2 small)",
                "task_description": "proportion of 'x' tokens at each position",
            },
        )
        self._test = TaskBatch(
            clean_tokens=things.test_data,
            corrupted_tokens=things.test_patch_data,
            correct_labels=things.test_labels,
            wrong_labels=None,
            metric=things.test_metrics["l2"],
            metadata={
                "source": "acdc.tracr_task.utils.get_all_tracr_things(proportion)",
                "model": "tracr-compiled tiny transformer",
            },
        )
