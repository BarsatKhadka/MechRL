"""Sanity test for CircuitReward on IOI.

Checks:
  1. begin_episode runs without error, faith ~= 0.75 (known IOI top-3k value)
  2. Cutting a canonical head -> delta_faith positive or near-zero
  3. terminal(all_alive) -> minimality=0 -> r_T=0
  4. terminal(all_cut)   -> faith=0     -> r_T=0
  5. terminal(top-3k)    -> both nonzero -> r_T > 0
"""

import sys
import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import build_graph, Prefilter, AblationEngine, CircuitReward


def main():
    print("Loading IOI task...")
    task = IOITask(num_examples=20, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)
    reward = CircuitReward(engine, sparsity_weight=0.001, step_budget=500)

    print("Computing prefilter top-3k...")
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)
    candidate_mask = pref.candidate_mask(3000)
    n_candidates = int(candidate_mask.sum().item())
    print(f"  candidates: {n_candidates}")

    # 1. begin_episode
    reward.begin_episode(candidate_mask)
    faith_start = reward._faith_before
    print(f"\n[1] begin_episode faith: {faith_start:.4f}  (expect ~0.75)")

    # 2. terminal(all_alive) -> minimality clipped to 0 (kept > n_candidates) -> 0
    all_alive = engine.all_alive_mask()
    r_alive = reward.terminal(all_alive)
    print(f"[2] terminal(all_alive): {r_alive:.4f}  (expect 0.0, minimality clipped to 0)")

    # 3. terminal(all_cut) -> faith=0 -> 0
    all_cut = engine.all_cut_mask()
    r_cut = reward.terminal(all_cut)
    print(f"[3] terminal(all_cut):   {r_cut:.4f}  (expect 0.0, faith<=0)")

    # 4. terminal(top-3k, all kept) -> minimality=0 -> 0  (agent did nothing)
    r_top3k_nocut = reward.terminal(candidate_mask)
    print(f"[4] terminal(top-3k, no cut): {r_top3k_nocut:.4f}  (expect 0.0, agent kept everything)")

    # 5. terminal(top-1k EAP-IG) -> kept=1000/3000, faith~0.52, minimality=0.67 -> nonzero
    top1k_mask = pref.candidate_mask(1000)
    r_pruned = reward.terminal(top1k_mask)
    faith_1k = float(engine.faithfulness(top1k_mask))
    kept_1k = int(top1k_mask.sum().item())
    min_1k = 1.0 - kept_1k / n_candidates
    print(f"[5] terminal(top-1k EAP-IG):  {r_pruned:.4f}  faith={faith_1k:.3f} minimality={min_1k:.3f}  (expect > 0)")

    # 6. One step: cut 100 bottom-ranked candidates (least important), check step reward
    reward.begin_episode(candidate_mask)  # reset faith tracking
    mask_step = candidate_mask.clone()
    cand_indices = candidate_mask.nonzero(as_tuple=True)[0]
    mask_step[cand_indices[-100:]] = False  # cut 100 least important (ranked last)
    r_step = reward.step(mask_step, valid_action=True)
    print(f"[6] step reward (cut 100 tail edges): {r_step:+.4f}  (expect small negative to small positive)")

    # 7. Invalid action penalty
    r_invalid = reward.step(mask_step, valid_action=False)
    print(f"[7] invalid action penalty: {r_invalid:+.4f}  (expect -0.01)")

    print("\nAll checks done.")


if __name__ == "__main__":
    main()
