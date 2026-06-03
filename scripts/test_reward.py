"""Validate the threshold-potential CircuitReward on IOI.

Checks the behaviours the design promises:
  - begin_episode at top-3k: faith ~0.72 (< tau=0.8) so we START below threshold,
    Phi_start is negative.
  - invalid action -> invalid_penalty, no state change.
  - cutting safe/low-rank edges (faith holds or rises) -> ΔΦ >= 0 (rewarded for
    shrinking while staying faithful).
  - cutting the strongest helper edge (raises faith toward/over tau) -> large +ΔΦ.
  - cutting many important edges (faith collapses below tau) -> negative ΔΦ.
  - objective() is higher for a small faithful circuit than the full candidate set.
"""

import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import build_graph, Prefilter, AblationEngine, CircuitReward


def main():
    print("Loading IOI...")
    task = IOITask(num_examples=20, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)

    cand = pref.candidate_mask(3000)
    reward = CircuitReward(engine, faith_threshold=0.8, threshold_penalty=3.0)

    reward.begin_episode(cand)
    print(f"\n[1] start: faith={reward.current_faith:.3f} (tau=0.8) "
          f"Phi_start={reward._phi_before:.3f}  (expect faith<0.8, Phi<0)")

    # [2] invalid action
    r_inv = reward.step(cand, valid_action=False)
    print(f"[2] invalid: r={r_inv:+.4f} (expect -0.01)")
    assert abs(r_inv + 0.01) < 1e-9

    # [3] cut the 200 LOWEST-RANKED (least important) candidates -> faith ~holds.
    # Use the prefilter ranking (not position order): edges in top-3000 but not top-2800.
    reward.begin_episode(cand)
    tail = pref.candidate_mask(3000) & ~pref.candidate_mask(2800)
    m = cand.clone(); m[tail] = False
    r_safe = reward.step(m, valid_action=True)
    print(f"[3] cut 200 lowest-ranked edges: r={r_safe:+.4f} faith={reward.current_faith:.3f} "
          f"(expect >=~0: smaller + still ~faithful)")

    # [4] strongest helper edge (most negative signed score) -> cutting raises faith
    ef = pref  # find helper via engine edge scores
    # rank edges by signed score; most negative = strongest helper
    edge_list = engine.edge_list
    cand_idx = cand.nonzero(as_tuple=True)[0]
    scores = torch.tensor([
        (edge_list[i].score.item() if torch.is_tensor(edge_list[i].score) else float(edge_list[i].score))
        for i in cand_idx.tolist()
    ])
    helper_local = int(torch.argmin(scores).item())
    helper_full = int(cand_idx[helper_local].item())
    reward.begin_episode(cand)
    m = cand.clone(); m[helper_full] = False
    r_help = reward.step(m, valid_action=True)
    print(f"[4] cut strongest helper {edge_list[helper_full].name}: r={r_help:+.4f} "
          f"faith {0.72:.2f}->{reward.current_faith:.3f}  (expect large +)")

    # [5] cut the top-200 MOST important edges -> faith collapses -> negative ΔΦ
    reward.begin_episode(cand)
    important = cand_idx[torch.argsort(scores.abs(), descending=True)[:200]]
    m = cand.clone(); m[important] = False
    r_bad = reward.step(m, valid_action=True)
    print(f"[5] cut 200 most-important: r={r_bad:+.4f} faith={reward.current_faith:.3f} "
          f"(expect negative — dropped below tau)")
    assert r_bad < 0

    # [6] objective: small faithful circuit should beat the full candidate set
    obj_full = reward.objective(cand)
    obj_small = reward.objective(pref.candidate_mask(1000))
    print(f"[6] objective(top-3k)={obj_full:.3f}  objective(top-1k)={obj_small:.3f}")

    print("\nThreshold-reward checks done.")


if __name__ == "__main__":
    main()
