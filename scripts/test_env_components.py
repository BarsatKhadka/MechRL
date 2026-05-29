"""Exercise the three env components (graph, prefilter, ablation) on IOI.

Sanity checks:
  1. build_graph produces 32k edges, 158 nodes.
  2. Prefilter top-K=3000 retains all 26 canonical IOI heads.
  3. AblationEngine.run_with_mask(all_alive) ≈ full clean model baseline.
  4. AblationEngine.run_with_mask(all_cut) ≈ corrupted prompt baseline.
  5. Random partial masks produce intermediate values.
"""

from __future__ import annotations

import torch

from acdc.ioi.utils import IOI_CIRCUIT
from mechrl.tasks import IOITask
from mechrl.env import build_graph, Prefilter, AblationEngine


def main():
    canonical = set()
    for heads in IOI_CIRCUIT.values():
        canonical.update(heads)

    print("=" * 70)
    print("STEP 1: build IOI task and graph")
    print("=" * 70)
    task = IOITask(num_examples=30, device="cpu")
    model = task.model
    graph = build_graph(model)
    print(f"  graph nodes:      {len(graph.nodes)}")
    print(f"  graph edges:      {len(graph.edges)}")
    print(f"  expected: 158 nodes, ~32,491 edges")

    print("\n" + "=" * 70)
    print("STEP 2: prefilter — compute attribution + top-K candidates")
    print("=" * 70)
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)
    print(f"  computed (or loaded from cache: {pref._cache_path().exists()})")

    for k in [500, 1000, 3000]:
        heads = pref.unique_heads_in_top_k(k)
        retained = canonical & heads
        print(f"  K={k:>5}: {len(heads)} unique heads,  {len(retained)}/{len(canonical)} canonical retained")

    candidate_mask_3k = pref.candidate_mask(3000)
    print(f"  candidate_mask(3000): {candidate_mask_3k.sum().item()} edges marked as candidates")

    print("\n" + "=" * 70)
    print("STEP 3: ablation engine — sanity tests")
    print("=" * 70)
    engine = AblationEngine(task, graph, batch_size=10)
    print(f"  engine: {engine.n_edges} edges total")

    full_baseline = engine.full_baseline()
    print(f"\n  Full baseline (no ablation): {full_baseline:.4f}")
    print(f"    (this is logit-diff of correct vs wrong on full clean model)")

    print("\n  Test 1: run_with_mask(all_alive) — should match full baseline")
    score_alive = engine.run_with_mask(engine.all_alive_mask())
    print(f"    run_with_mask(all alive):  {score_alive:.4f}")
    print(f"    full baseline:             {full_baseline:.4f}")
    diff = abs(score_alive - full_baseline)
    if diff < 0.01:
        print(f"    [PASS] difference {diff:.6f} < 0.01")
    else:
        print(f"    [FAIL] difference {diff:.6f} >= 0.01")

    print("\n  Test 2: run_with_mask(all_cut) — should be much lower (no signal)")
    score_cut = engine.run_with_mask(engine.all_cut_mask())
    print(f"    run_with_mask(all cut):    {score_cut:.4f}")
    print(f"    full baseline:             {full_baseline:.4f}")
    if score_cut < full_baseline - 1.0:
        print(f"    [PASS] all-cut score is much lower than baseline")
    else:
        print(f"    [WARN] all-cut score not much below baseline (suspicious)")

    print("\n  Test 3: run_with_mask(random 50% alive) — should be between")
    torch.manual_seed(0)
    random_mask = torch.rand(engine.n_edges) < 0.5
    score_random = engine.run_with_mask(random_mask)
    print(f"    run_with_mask(50% random): {score_random:.4f}")
    print(f"    (expected: somewhere between {score_cut:.2f} and {full_baseline:.2f})")
    if score_cut <= score_random <= full_baseline + 0.5:
        print(f"    [PASS] intermediate value")
    else:
        print(f"    [WARN] out of expected range")

    print("\n" + "=" * 70)
    print("STEP 4: candidate-only mask — keep only top-K edges alive")
    print("=" * 70)
    print("  This is what the agent's STARTING mask looks like at episode reset:")
    print("  all candidate edges alive, everything outside top-K kept alive (frozen).")
    print("  We test: top-K=3000 alive + everything else alive = full model = full baseline")
    all_alive = engine.all_alive_mask()
    score_full = engine.run_with_mask(all_alive)
    print(f"    score with all 32k alive: {score_full:.4f}")
    print(f"    full baseline:            {full_baseline:.4f}")
    print(f"    (same as Test 1)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
