"""Env-correctness checks that must pass before training a policy on it.

These catch the silent bugs that would corrupt RL training:
  1. DETERMINISM   — same seed + same actions => identical rewards (twice).
                     Proves no state leaks between episodes via the shared graph.
  2. KILL == CUTs  — killing a node lands on the SAME faithfulness as cutting
                     each of its edges one by one. Proves the macro action is
                     consistent with single cuts.
  3. MASKING       — invalid actions are penalized and leave state untouched.
  4. RESET CLEAN   — after reset, all candidates alive, faith back at start.
"""

import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import CircuitEnv, TaskBundle


def approx(a, b, tol=1e-5):
    return abs(a - b) <= tol


def main():
    print("Building IOI bundle...")
    task = IOITask(num_examples=20, device="cpu")
    bundle = TaskBundle.build(task, k=3000)
    K = bundle.n_candidates
    M = len(bundle.parent_names)
    STOP = K + M

    # ---- 1. DETERMINISM ----
    print("\n[1] Determinism: same seed + actions -> identical rewards")
    actions = [K - 1, K - 5, K - 9, K - 13, STOP]

    def run(seed):
        env = CircuitEnv([bundle], step_budget=50, seed=seed)
        env.reset(bundle_idx=0)
        rs, faiths = [], []
        for a in actions:
            _, r, done, info = env.step(a)
            rs.append(round(r, 6))
            faiths.append(round(info["faith"], 6))
            if done:
                break
        return rs, faiths

    r1, f1 = run(0)
    r2, f2 = run(0)
    print(f"  run1 rewards: {r1}")
    print(f"  run2 rewards: {r2}")
    assert r1 == r2, "NON-DETERMINISTIC rewards across identical runs!"
    assert f1 == f2, "NON-DETERMINISTIC faith across identical runs!"
    print("  PASS - identical across runs (no state leak)")

    # ---- 2. KILL == sequence of CUTs ----
    print("\n[2] KILL(node) == cutting its edges one by one")
    # pick a small parent group so the one-by-one path is fast
    sizes = [(m, int(g.numel())) for m, g in enumerate(bundle.parent_groups)]
    small = [ms for ms in sizes if 2 <= ms[1] <= 8]
    m_pick, sz = sorted(small, key=lambda x: x[1])[0]
    group = bundle.parent_groups[m_pick].tolist()
    print(f"  target node '{bundle.parent_names[m_pick]}' with {sz} edges: {group}")

    # path A: one KILL
    envA = CircuitEnv([bundle], step_budget=50, seed=0)
    envA.reset(bundle_idx=0)
    _, _, _, infoA = envA.step(K + m_pick)
    faith_kill = infoA["faith"]
    mask_kill = envA.mask.clone()

    # path B: cut each edge individually
    envB = CircuitEnv([bundle], step_budget=50, seed=0)
    envB.reset(bundle_idx=0)
    faith_cuts = None
    for e in group:
        _, _, _, infoB = envB.step(e)
        faith_cuts = infoB["faith"]
    mask_cuts = envB.mask.clone()

    print(f"  faith after KILL:        {faith_kill:.6f}")
    print(f"  faith after {sz} CUTs:    {faith_cuts:.6f}")
    assert torch.equal(mask_kill, mask_cuts), "KILL and CUTs produced different masks!"
    assert approx(faith_kill, faith_cuts), "KILL and CUTs gave different faithfulness!"
    print("  PASS - identical final mask and faithfulness")

    # ---- 3. MASKING: invalid action ----
    print("\n[3] Invalid action: penalized, state untouched")
    env = CircuitEnv([bundle], step_budget=50, seed=0, invalid_penalty=-0.01)
    env.reset(bundle_idx=0)
    env.step(K - 1)                       # cut edge K-1
    alive_before = env.alive.sum().item()
    faith_before = env.reward.current_faith
    _, r, _, info = env.step(K - 1)       # cut it AGAIN -> invalid
    alive_after = env.alive.sum().item()
    print(f"  reward={r:+.4f} (expect -0.01), reason={info['reason']}")
    assert approx(r, -0.01), "invalid action not penalized correctly"
    assert alive_before == alive_after, "invalid action changed alive state!"
    assert approx(faith_before, env.reward.current_faith), "invalid action changed faith!"
    print("  PASS - penalty applied, no state change")

    # ---- 4. RESET CLEAN ----
    print("\n[4] Reset returns clean initial state")
    env = CircuitEnv([bundle], step_budget=50, seed=0)
    obs = env.reset(bundle_idx=0)
    for a in [K - 1, K - 2, K + m_pick]:
        env.step(a)
    obs2 = env.reset(bundle_idx=0)
    print(f"  all alive after reset: {bool(env.alive.all())}")
    print(f"  faith back to start:   {env.reward.current_faith:.4f} (start={env.faith_start:.4f})")
    assert bool(env.alive.all()), "reset did not restore all-alive"
    assert approx(env.reward.current_faith, env.faith_start), "reset did not restore faith"
    print("  PASS - clean reset")

    print("\nAll env-correctness checks passed.")


if __name__ == "__main__":
    main()
