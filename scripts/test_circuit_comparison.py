"""Compare different masks on IOI to find what actually achieves high faithfulness.

Tests several mask strategies:
  1. Full alive (no cuts)
  2. All cut (max corruption)
  3. EAP-IG top-3000 (our agent's starting candidate pool)
  4. EAP-IG top-1000
  5. EAP-IG top-500
  6. Wang et al. precise circuit (channel-specific)
  7. Canonical-touching (loose superset)

Goal: understand what level of edge inclusion gives ~80%+ faithfulness so
we know what target the agent is shooting for.
"""

from __future__ import annotations

import torch

from mechrl.tasks import IOITask
from mechrl.env import build_graph, Prefilter, AblationEngine


def main():
    task = IOITask(num_examples=30, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)

    full = engine.full_baseline()
    all_cut = engine.run_with_mask(engine.all_cut_mask())
    print(f"\nFull baseline:        {full:+.4f}")
    print(f"All cut (worst):      {all_cut:+.4f}")
    print(f"Maximum possible drop: {abs(full - all_cut):.4f}")
    print()
    print(f"  {'mask':>45} | {'edges':>6} | {'score':>9} | {'faithfulness':>12}")
    print(f"  {'-'*45} | {'-'*6} | {'-'*9} | {'-'*12}")

    def report(name, mask):
        n = mask.sum().item()
        s = engine.run_with_mask(mask)
        f = s / full if full != 0 else float("nan")
        print(f"  {name:>45} | {n:>6} | {s:>+9.4f} | {f:>11.2%}")

    # Baselines
    report("full alive", engine.all_alive_mask())
    report("all cut", engine.all_cut_mask())

    # EAP-IG top-K candidate masks
    for k in [500, 1000, 2000, 3000, 5000, 10000, 20000]:
        report(f"EAP-IG top-{k}", pref.candidate_mask(k))


if __name__ == "__main__":
    main()
