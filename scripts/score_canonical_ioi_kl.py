"""What is the KL-faithfulness of Wang et al.'s canonical IOI circuit?

The ACDC paper says these edges "are responsible for IOI" and reports ~87%
faithfulness — but under LOGIT-DIFF, not our KL metric. Our whole project now
measures faithfulness as normalized KL (faith = 1 - KL(circuit)/KL(all-cut)).
So the canonical circuit has never been scored on OUR axis.

This puts the canonical circuit on the SAME scale as our agent (which keeps
~1450 edges at KL-faith ~0.9), so the size-vs-faith comparison is apples-to-
apples. Reuses the verified precise-circuit mask builder from
test_ioi_precise_circuit.py.

Run (cheap: one model load + ~5 forward passes):
    python -m scripts.score_canonical_ioi_kl
"""

from __future__ import annotations

import torch

from mechrl.tasks import IOITask
from mechrl.env import build_graph, AblationEngine

# Verified mapping: Wang et al.'s precise channel-specific circuit -> our edges.
from scripts.test_ioi_precise_circuit import build_precise_mask, build_random_mask


def main():
    print("Loading IOI task + graph (num_examples=20)...", flush=True)
    task = IOITask(num_examples=20, device="cpu")
    graph = build_graph(task.model)

    # The one change vs test_ioi_precise_circuit.py: KL metric, not logit-diff.
    engine = AblationEngine(task, graph, metric_type="kl")

    n_total = engine.n_edges
    print(f"Total edges in our graph: {n_total}", flush=True)

    # KL baselines (full model -> KL 0 by construction; all-cut -> max KL)
    kl_full = engine.full_baseline()        # ~0.0
    kl_cut = engine.corrupted_baseline()    # KL(full || all-cut)
    print(f"KL full-model (reference): {kl_full:.4f}", flush=True)
    print(f"KL all-cut (worst):        {kl_cut:.4f}", flush=True)

    # --- The canonical circuit ---
    print("\nBuilding canonical (Wang et al.) circuit mask...", flush=True)
    canon = build_precise_mask(engine)
    n_canon = int(canon.sum().item())
    kl_canon = engine.run_with_mask(canon)
    faith_canon = engine.faithfulness(canon)
    print(f"  canonical edges (in our graph): {n_canon}", flush=True)
    print(f"  KL(circuit):                    {kl_canon:.4f}", flush=True)
    print(f"  KL-faith = 1 - KL/KL_cut:       {faith_canon:.4f}", flush=True)

    # --- Random circuits of the SAME size, for context ---
    print(f"\nRandom circuits of same size ({n_canon} edges, 3 trials):", flush=True)
    faiths = []
    for seed in range(3):
        rmask = build_random_mask(n_total, n_canon, seed=seed)
        fr = engine.faithfulness(rmask)
        faiths.append(fr)
        print(f"  trial {seed}: KL-faith {fr:.4f}", flush=True)
    print(f"  mean random KL-faith: {sum(faiths)/len(faiths):.4f}", flush=True)

    print("\n=== SUMMARY (all on OUR KL-faith axis) ===", flush=True)
    print(f"  canonical (Wang et al.): {n_canon:>5} edges @ KL-faith {faith_canon:.3f}", flush=True)
    print(f"  our agent (tau=0.9):     ~1450 edges @ KL-faith ~0.86-0.90  (prior run)", flush=True)
    print(f"  random (same size):       {n_canon:>5} edges @ KL-faith {sum(faiths)/len(faiths):.3f}", flush=True)


if __name__ == "__main__":
    main()
