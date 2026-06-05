"""Diagnostic: score the HEAD-LEVEL canonical IOI masks under KL.

The precise channel-specific mask gave KL-faith ~0.007 (== random), which looks
like a mapping bug. The head-level "touching" mask is the one that PASSED the
Gate-4 validation under logit-diff. If it scores high under KL, the engine is
fine and the precise reconstruction is broken. If it ALSO scores ~0, the issue
is in the KL ablation / our understanding.
"""

from __future__ import annotations

from mechrl.tasks import IOITask
from mechrl.env import build_graph, AblationEngine

from scripts.test_canonical_circuits import (
    IOI_CANONICAL,
    build_canonical_mask,
    build_canonical_touching_mask,
    build_random_mask,
)


def main():
    print("Loading IOI (num_examples=20)...", flush=True)
    task = IOITask(num_examples=20, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph, metric_type="kl")

    kl_cut = engine.corrupted_baseline()
    print(f"KL all-cut: {kl_cut:.4f}\n", flush=True)

    for label, builder in [
        ("strict (both endpoints canonical)", build_canonical_mask),
        ("touching (>=1 endpoint canonical)", build_canonical_touching_mask),
    ]:
        mask = builder(engine, IOI_CANONICAL)
        n = int(mask.sum().item())
        faith = engine.faithfulness(mask)
        print(f"{label}: {n:>6} edges  KL-faith {faith:.4f}", flush=True)
        rmask = build_random_mask(engine, n, seed=0)
        print(f"    random same-size: KL-faith {engine.faithfulness(rmask):.4f}", flush=True)


if __name__ == "__main__":
    main()
