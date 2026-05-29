"""Task interface for circuit-discovery RL environment.

A Task bundles together everything the env needs to evaluate one kind of
behavior in a transformer:

- a clean / corrupted prompt pair (for ABC patching)
- the answer tokens (for logit-difference metrics)
- a metric function (logits, batch) -> per-prompt score
- the underlying model the task runs on

We wrap ACDC's task generators (e.g. get_all_ioi_things) and expose them
behind a uniform interface so the env doesn't need to know task internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from transformer_lens import HookedTransformer


@dataclass
class TaskBatch:
    """One batch of prompts ready to feed to the model.

    clean_tokens     : [batch, seq_len]    real prompts
    corrupted_tokens : [batch, seq_len]    structurally-matched corrupted prompts
    correct_labels   : [batch]             answer token id per prompt (e.g. IO for IOI)
    wrong_labels     : [batch] | None      distractor token id per prompt (e.g. S for IOI)
    metric           : callable(logits) -> [batch]   per-prompt task score
    metadata         : dict                anything extra worth keeping (templates, raw strings, etc.)
    """

    clean_tokens: torch.Tensor
    corrupted_tokens: torch.Tensor
    correct_labels: torch.Tensor
    wrong_labels: Optional[torch.Tensor]
    metric: Callable[[torch.Tensor], torch.Tensor]
    metadata: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.clean_tokens.shape[0]

    @property
    def seq_len(self) -> int:
        return self.clean_tokens.shape[1]


class Task(ABC):
    """A circuit-discovery task on some transformer model."""

    name: str

    def __init__(self, num_examples: int = 64, device: str = "cpu", seed: int = 0):
        self.num_examples = num_examples
        self.device = device
        self.seed = seed
        self._model: Optional[HookedTransformer] = None
        self._validation: Optional[TaskBatch] = None
        self._test: Optional[TaskBatch] = None

    @property
    def model(self) -> HookedTransformer:
        if self._model is None:
            self._build()
        return self._model

    def validation_batch(self) -> TaskBatch:
        if self._validation is None:
            self._build()
        return self._validation

    def test_batch(self) -> TaskBatch:
        if self._test is None:
            self._build()
        return self._test

    @abstractmethod
    def _build(self) -> None:
        """Populate self._model, self._validation, self._test by wrapping ACDC."""
        ...

    def eap_labels(self, batch: "TaskBatch") -> torch.Tensor:
        """Return per-prompt (correct_id, wrong_id) pairs for the logit-diff metric
        that EAP-IG attribution expects. Default: use correct_labels and
        wrong_labels from the batch if present; otherwise use correct + (correct+1).

        Override in subclasses for task-specific logic (e.g. IOI extracting the
        subject name).
        """
        n = batch.batch_size
        if batch.wrong_labels is not None:
            wrong = batch.wrong_labels[:n]
        elif batch.correct_labels is not None:
            wrong = (batch.correct_labels[:n] + 1) % 50257  # arbitrary distractor
        else:
            return None
        return torch.stack([batch.correct_labels[:n], wrong], dim=1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, num_examples={self.num_examples})"
