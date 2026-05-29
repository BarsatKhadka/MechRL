"""Test that cutting negative heads (L10.H7, L11.H10) INCREASES faithfulness.

If true, this confirms:
  - The agent can improve from the 75% top-3000 baseline
  - There are "wrong" edges in the candidate set whose removal helps
  - The reward landscape is non-monotonic in mask size
"""

from __future__ import annotations

import torch

from mechrl.tasks import IOITask
from mechrl.env import build_graph, Prefilter, AblationEngine


def edges_touching_head(engine, layer, head):
    """Return indices of edges where parent or child is the given head."""
    target_name = f"a{layer}.h{head}"
    indices = []
    for i, edge in enumerate(engine.edge_list):
        if edge.parent.name == target_name or edge.child.name == target_name:
            indices.append(i)
    return indices


def main():
    task = IOITask(num_examples=30, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)

    full = engine.full_baseline()
    print(f"\nFull baseline:        {full:+.4f}\n")

    # Start at top-3000
    base_mask = pref.candidate_mask(3000)
    base_score = engine.run_with_mask(base_mask)
    base_faith = base_score / full
    base_alive = base_mask.sum().item()
    print(f"Top-3000 baseline:    {base_score:+.4f}  ({base_faith:.2%} faithfulness, {base_alive} edges alive)")

    print("\nNow cut edges TOUCHING specific heads (one at a time):")
    print(f"  {'head':>10} | {'role':>20} | {'edges cut':>9} | {'after score':>11} | {'after faith':>11} | {'delta':>8}")
    print(f"  {'-'*10} | {'-'*20} | {'-'*9} | {'-'*11} | {'-'*11} | {'-'*8}")

    test_heads = [
        ((10, 7), "negative name mover"),
        ((11, 10), "negative name mover"),
        ((9, 6), "name mover"),
        ((9, 9), "name mover"),
        ((10, 0), "name mover"),
        ((5, 5), "induction"),
        ((7, 3), "S-inhibition"),
        ((0, 1), "duplicate token"),
    ]

    for (layer, head), role in test_heads:
        mask = base_mask.clone()
        edges_to_cut = edges_touching_head(engine, layer, head)
        # Only cut edges that are currently alive in the base mask
        cuts_in_base = [i for i in edges_to_cut if mask[i]]
        for i in cuts_in_base:
            mask[i] = False
        score = engine.run_with_mask(mask)
        faith = score / full
        delta = faith - base_faith
        sign = "+" if delta > 0 else ""
        print(f"  L{layer:>2}.H{head:<2}    | {role:>20} | {len(cuts_in_base):>9} | "
              f"{score:>+11.4f} | {faith:>10.2%} | {sign}{delta:>+7.2%}")

    print("\nINTERPRETATION:")
    print("  - Negative deltas: cutting this head HURTS → agent should keep it")
    print("  - Positive deltas: cutting this head HELPS → agent should cut it (free improvement)")
    print("  - Near-zero: cutting doesn't matter → agent can cut for free (minimality bonus)")
    print()
    print("  If L10.H7 (negative head) gives a POSITIVE delta, the 75% floor isn't a ceiling.")


if __name__ == "__main__":
    main()
