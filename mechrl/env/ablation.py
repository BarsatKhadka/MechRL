"""AblationEngine — run the model with an arbitrary edge mask applied.

Wraps EAP-IG's evaluate_graph machinery so the rest of the env can just call
`engine.run_with_mask(mask)` to get logits-or-metric back.

Per call:
  - Sets graph.in_graph from the boolean mask
  - Runs evaluate_graph (which installs hooks and runs the forward pass with
    corrupted activations swapped in at cut edges)
  - Returns the metric value (or raw logits if requested)
"""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph

from mechrl.tasks.base import Task, TaskBatch


# Reuse the same collate / dataset format used by the prefilter
def _collate(xs):
    clean, corrupted, labels = zip(*xs)
    if labels[0] is None:
        return list(clean), list(corrupted), None
    return list(clean), list(corrupted), torch.stack(labels)


class _TaskEapDataset(Dataset):
    def __init__(self, task: Task, split: str = "validation"):
        batch = task.validation_batch() if split == "validation" else task.test_batch()
        model = task.model
        self.clean_strs = [model.to_string(batch.clean_tokens[i]) for i in range(batch.batch_size)]
        self.corrupt_strs = [model.to_string(batch.corrupted_tokens[i]) for i in range(batch.batch_size)]
        self.labels = task.eap_labels(batch)

    def __len__(self):
        return len(self.clean_strs)

    def __getitem__(self, i):
        lab = self.labels[i] if self.labels is not None else None
        return self.clean_strs[i], self.corrupt_strs[i], lab


def _logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    last = logits[idx, input_length - 1]
    labels = labels.to(last.device)
    good_bad = torch.gather(last, -1, labels)
    diff = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        diff = -diff
    if mean:
        diff = diff.mean()
    return diff


def _wrap_task_metric(task: Task) -> Callable:
    """Adapter: take task.validation_batch().metric (which expects only logits)
    and wrap it into the 4-arg signature eap-ig's evaluate_graph expects.

    The task's metric is called with the full logits tensor; the input_length
    and labels arguments are ignored (most task metrics handle position
    indexing internally).
    """
    task_metric = task.validation_batch().metric

    def adapter(logits, clean_logits, input_length, labels):
        result = task_metric(logits)
        if torch.is_tensor(result):
            if result.dim() == 0:
                return result
            return result.mean()
        return torch.tensor(float(result))

    return adapter


class AblationEngine:
    """Run GPT-2 with an arbitrary edge mask, return the metric value.

    Usage:
        engine = AblationEngine(task, graph, batch_size=10)
        full_score = engine.run_with_mask(torch.ones(n_edges, dtype=torch.bool))
        # ^ all edges alive → should match the clean model baseline
        zero_score = engine.run_with_mask(torch.zeros(n_edges, dtype=torch.bool))
        # ^ all edges cut → should match the corrupted-prompt baseline
    """

    def __init__(self, task: Task, graph: Graph, batch_size: Optional[int] = None,
                 use_task_metric: bool = True):
        """
        Parameters
        ----------
        task : Task
            The task to ablate against.
        graph : Graph
            The pre-built EAP-IG computational graph.
        batch_size : int
            Batch size for forward passes.
        use_task_metric : bool
            If True, wraps task.validation_batch().metric (which takes only
            logits) into the eap-ig 4-arg metric signature. This uses each
            task's NATURAL metric (logit-diff for IOI, prob-diff for greater-
            than, KL for docstring etc).
            If False, falls back to logit-diff using eap_labels.
        """
        self.task = task
        self.graph = graph
        # Default: use the full task batch as one DataLoader batch. This is
        # required when the task's metric is partialed with batch-size-specific
        # labels (IOI, greater-than). Pass batch_size explicitly to override.
        if batch_size is None:
            batch_size = task.num_examples
        self.batch_size = batch_size

        ds = _TaskEapDataset(task)
        self.dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=_collate)

        # Build an ordered list of edges so masks can be indexed consistently
        self.edge_list = list(graph.edges.values())
        self.n_edges = len(self.edge_list)

        # Metric: either task's natural metric or fallback logit-diff
        if use_task_metric:
            self.metric = _wrap_task_metric(task)
        else:
            self.metric = partial(_logit_diff, loss=False, mean=True)

        # Precompute baselines (cached for sanity tests)
        self._full_baseline: Optional[float] = None
        self._corrupted_baseline: Optional[float] = None

        # Per-task cache of the corrupted forward pass. The corrupted activations
        # are identical on every run_with_mask call (they don't depend on the mask),
        # so we compute them once and reuse — skipping one of the two GPT-2 forwards
        # per step. Built lazily on the first patching call. Reset via reset_cache().
        self.use_corrupted_cache = True
        self._corrupted_cache: list = []

    # ---- Baselines (independent of mask) ----

    def full_baseline(self) -> float:
        """Score of the clean model with NO ablation."""
        if self._full_baseline is None:
            score = evaluate_baseline(self.task.model, self.dataloader, self.metric).mean().item()
            self._full_baseline = score
        return self._full_baseline

    def corrupted_baseline(self) -> float:
        """Score with ALL edges cut (model runs on corrupted activations everywhere)."""
        if self._corrupted_baseline is None:
            self._corrupted_baseline = self.run_with_mask(self.all_cut_mask())
        return self._corrupted_baseline

    def faithfulness(self, mask: torch.Tensor) -> float:
        """Normalized faithfulness in [0, 1] regardless of task's metric sign/scale.

        formula:  (score - cut_baseline) / (full_baseline - cut_baseline)
            1.0 = matches full model exactly
            0.0 = same as all-cut (no useful information preserved)
            negative = catastrophic (worse than all-cut)
        """
        score = self.run_with_mask(mask)
        full = self.full_baseline()
        cut = self.corrupted_baseline()
        denom = full - cut
        if abs(denom) < 1e-8:
            return 0.0  # degenerate task — full ≈ cut means no signal exists
        return (score - cut) / denom

    # ---- Main mask-driven evaluation ----

    def _apply_mask(self, mask: torch.Tensor) -> None:
        """Set graph.in_graph from the boolean mask."""
        assert mask.shape == (self.n_edges,), f"mask shape {mask.shape} != {self.n_edges}"
        for i, edge in enumerate(self.edge_list):
            edge.in_graph = bool(mask[i].item())

    def run_with_mask(self, mask: torch.Tensor, intervention: str = "patching") -> float:
        """Run the model with the given mask applied, return mean metric value.

        mask[i] = True  → edge i is alive (uses clean activation)
        mask[i] = False → edge i is cut (replaced with intervention's value)

        intervention:
          - 'patching' (default): cut edges replaced with corrupted-prompt activations
          - 'mean': cut edges replaced with mean activation over the dataset
          - 'mean-positional': cut edges replaced with per-position mean activation
          - 'zero': cut edges replaced with zeros (off-manifold, not recommended)
        """
        self._apply_mask(mask)
        # Only 'patching' has a mask-independent corrupted pass worth caching.
        cache = self._corrupted_cache if (self.use_corrupted_cache and intervention == "patching") else None
        return evaluate_graph(
            self.task.model,
            self.graph,
            self.dataloader,
            self.metric,
            intervention=intervention,
            intervention_dataloader=self.dataloader if "mean" in intervention else None,
            quiet=True,
            corrupted_cache=cache,
        ).mean().item()

    def reset_cache(self) -> None:
        """Drop the cached corrupted pass (e.g. if the task's data changes)."""
        self._corrupted_cache = []

    # ---- Convenience methods ----

    def all_alive_mask(self) -> torch.Tensor:
        return torch.ones(self.n_edges, dtype=torch.bool)

    def all_cut_mask(self) -> torch.Tensor:
        return torch.zeros(self.n_edges, dtype=torch.bool)
