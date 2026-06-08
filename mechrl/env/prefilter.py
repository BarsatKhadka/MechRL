"""Prefilter — EAP-IG attribution + top-K candidate edge selection.

Per task: compute attribution scores for every edge, then pick the top-K
edges as the candidate set the agent gets to decide about.

Computation is cached on disk per (task_name, num_examples, ig_steps) so
re-running with the same parameters is instant.
"""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eap.attribute import attribute
from eap.graph import Graph

from mechrl.tasks.base import Task


# ----- eap-ig dataloader format -----

def _collate(xs):
    clean, corrupted, labels = zip(*xs)
    if labels[0] is None:
        return list(clean), list(corrupted), None
    return list(clean), list(corrupted), torch.stack(labels)


class _TaskEapDataset(Dataset):
    """Adapter from a mechrl Task to the (clean_str, corrupt_str, label_pair)
    format eap-ig's dataloader expects."""

    def __init__(self, task: Task):
        batch = task.validation_batch()
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
    """Standard logit-diff loss: logit[correct] - logit[wrong] at final position.
    Used as fallback when task doesn't provide its own metric."""
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


def _kl_attr_metric(logits, clean_logits, input_length, labels):
    """KL( full-model || circuit ) at the prediction position, for EAP-IG attribution.

    Ranks candidate edges by how much they matter to reproducing the full model's
    DISTRIBUTION -- i.e. exactly the KL faithfulness the agent is scored on -- instead
    of logit-diff (two tokens). This fixes the prefilter<->engine mismatch that capped
    diffuse tasks (docstring) low: logit-diff-important edges are not the same as
    distribution-reproducing edges.

    Uses `clean_logits` (the full model's clean output) as the reference -- attribute()
    computes it for us (eap/attribute.py:90, comment names KL as the case), so this is
    consistent with AblationEngine's KL metric (same model, same clean prompts).
    KL >= 0, lower = better; top-K selection is by |score| so the sign is irrelevant.
    """
    bs = logits.size(0)
    idx = torch.arange(bs, device=logits.device)
    pos = (input_length - 1).to(logits.device).long()          # prediction position
    circ = logits[idx, pos].float()
    ref = clean_logits[idx, pos].float()
    logp_circ = F.log_softmax(circ, dim=-1)
    logp_ref = F.log_softmax(ref, dim=-1)
    kl = (logp_ref.exp() * (logp_ref - logp_circ)).sum(-1)      # KL per example
    return kl.mean()


def _wrap_task_metric(task: Task) -> Callable:
    """Wrap task.validation_batch().metric (one-arg, takes logits) into the
    4-arg signature eap-ig's attribute() expects. Each task's natural metric
    is already in 'lower = better' form, so we use it as-is for IG attribution.
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


class Prefilter:
    """Compute and cache EAP-IG scores for a task; expose top-K candidate edges.

    Usage:
        pref = Prefilter(task, graph, ig_steps=5)
        pref.compute(batch_size=10)             # runs attribution, caches scores
        candidates = pref.top_k_edges(3000)     # returns list of Edge objects
        mask = pref.candidate_mask(3000)        # boolean tensor over all edges
    """

    def __init__(
        self,
        task: Task,
        graph: Graph,
        ig_steps: int = 5,
        cache_dir: Optional[Path] = None,
        use_task_metric: bool = True,
        metric_type: Optional[str] = None,
    ):
        self.task = task
        self.graph = graph
        self.ig_steps = ig_steps
        self.use_task_metric = use_task_metric
        # metric_type="kl" -> rank candidates by KL attribution (matches the engine).
        # None -> task's natural metric (logit-diff) as before. Opt-in so existing
        # logit-diff-attributed runs (the locked single-task IOI) stay reproducible.
        self.metric_type = metric_type
        self.cache_dir = cache_dir or (
            Path(__file__).resolve().parents[2] / "prefilter_cache"
        )
        self.cache_dir.mkdir(exist_ok=True)
        self._scored = False

    def _cache_path(self) -> Path:
        if self.metric_type == "kl":
            suffix = "_kl"
        elif self.use_task_metric:
            suffix = "_taskmetric"
        else:
            suffix = "_logitdiff"
        return (
            self.cache_dir
            / f"{self.task.name}_n{self.task.num_examples}_ig{self.ig_steps}{suffix}.pt"
        )

    def compute(self, batch_size: int = 10, force: bool = False) -> None:
        """Run EAP-IG attribution. Writes scores into self.graph.scores.
        Caches the scores to disk so subsequent calls are instant.
        """
        cache_path = self._cache_path()
        if cache_path.exists() and not force:
            scores = torch.load(cache_path, map_location=self.task.device)
            assert scores.shape == self.graph.scores.shape, (
                f"Cached scores shape {scores.shape} != graph {self.graph.scores.shape}"
            )
            self.graph.scores[:] = scores
            self._scored = True
            return

        ds = _TaskEapDataset(self.task)
        # Must match AblationEngine: use one batch covering all examples so
        # task metrics with batch-sized labels (and the KL reference) work correctly.
        if self.metric_type == "kl":
            effective_batch_size = self.task.num_examples
            metric_fn = _kl_attr_metric
        elif self.use_task_metric:
            effective_batch_size = self.task.num_examples
            metric_fn = _wrap_task_metric(self.task)
        else:
            effective_batch_size = batch_size
            metric_fn = partial(_logit_diff, loss=True, mean=True)
        loader = DataLoader(ds, batch_size=effective_batch_size, collate_fn=_collate)
        attribute(
            self.task.model,
            self.graph,
            loader,
            metric_fn,
            method="EAP-IG-inputs",
            ig_steps=self.ig_steps,
        )
        # Cache for future runs
        torch.save(self.graph.scores.clone(), cache_path)
        self._scored = True

    def _ranked_edges(self):
        """All edges sorted by |score| descending."""
        if not self._scored:
            raise RuntimeError("Call compute() before reading scores.")
        return sorted(
            self.graph.edges.values(),
            key=lambda e: -abs(
                e.score.item() if torch.is_tensor(e.score) else float(e.score)
            ),
        )

    def top_k_edges(self, k: int) -> List:
        """Return the top-K edges by absolute score."""
        return self._ranked_edges()[:k]

    def candidate_mask(self, k: int) -> torch.Tensor:
        """Boolean tensor of length n_edges; True for edges in the top-K."""
        n_edges = len(self.graph.edges)
        mask = torch.zeros(n_edges, dtype=torch.bool)
        edge_list = list(self.graph.edges.values())
        edge_to_idx = {id(e): i for i, e in enumerate(edge_list)}
        for edge in self.top_k_edges(k):
            mask[edge_to_idx[id(edge)]] = True
        return mask

    def unique_heads_in_top_k(self, k: int) -> set:
        """Return set of (layer, head) appearing in any of the top-K edges."""
        heads = set()
        for edge in self.top_k_edges(k):
            for node in (edge.parent, edge.child):
                name = node.name
                if name.startswith("a"):
                    try:
                        l, h = name.split(".")
                        heads.add((int(l[1:]), int(h[1:])))
                    except (ValueError, IndexError):
                        pass
        return heads
