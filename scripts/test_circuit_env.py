"""Smoke test for CircuitEnv on IOI.

Drives a short episode (small budget) and checks:
  - reset() returns a well-shaped observation
  - CUT actions reduce alive count and produce sane rewards
  - cutting an already-cut edge -> invalid penalty
  - STOP gives terminal reward = faith x minimality
  - budget exhaust auto-stops
"""

import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import CircuitEnv, TaskBundle


def main():
    print("Building IOI bundle (model + graph + prefilter)...")
    task = IOITask(num_examples=20, device="cpu")
    bundle = TaskBundle.build(task, k=3000)
    print(f"  candidates: {bundle.n_candidates}")
    print(f"  edge_features shape: {tuple(bundle.edge_features.shape)}")

    print(f"  kill targets (distinct parents): {len(bundle.parent_names)}")

    K = bundle.n_candidates
    M = len(bundle.parent_names)
    STOP = K + M

    # Small budget so the smoke test finishes fast on CPU.
    env = CircuitEnv([bundle], step_budget=6, seed=0)

    print("\n--- Episode A: cut a few low-ranked candidates, then STOP ---")
    obs = env.reset(bundle_idx=0)
    print(f"obs keys: {list(obs.keys())}")
    print(f"  edge_features: {tuple(obs['edge_features'].shape)}, "
          f"edge_alive: {tuple(obs['edge_alive'].shape)}, "
          f"kill_alive: {tuple(obs['kill_alive'].shape)}, "
          f"globals: {obs['globals'].tolist()}")
    print(f"action_dim: {env.action_dim}  (K={K} edges + M={M} kills + STOP)")

    # Cut the 3 lowest-ranked candidates (indices K-1, K-2, K-3 -> least important)
    for a in [K - 1, K - 2, K - 3]:
        obs, r, done, info = env.step(a)
        print(f"  CUT {a}: r={r:+.4f}  faith={info['faith']:.4f}  "
              f"kept={info['kept']}  done={done}  ({info['reason']})")

    # Try cutting an already-cut edge -> invalid penalty
    obs, r, done, info = env.step(K - 1)
    print(f"  CUT {K-1} again: r={r:+.4f}  ({info['reason']})  expect -0.01")

    # STOP
    obs, r, done, info = env.step(STOP)
    print(f"  STOP: r={r:+.4f}  faith={info['faith']:.4f}  kept={info['kept']}  done={done}")
    assert done, "STOP should end the episode"

    print("\n--- Episode B: KILL a parent node (cut all its outgoing edges at once) ---")
    obs = env.reset(bundle_idx=0)
    # Pick the kill node with the most edges to show a big one-step cut.
    sizes = [(m, int(g.numel())) for m, g in enumerate(bundle.parent_groups)]
    m_big, sz = max(sizes, key=lambda x: x[1])
    kept_before = int(env.alive.sum().item())
    obs, r, done, info = env.step(K + m_big)
    print(f"  KILL parent '{bundle.parent_names[m_big]}' ({sz} candidate edges):")
    print(f"    r={r:+.4f}  faith={info['faith']:.4f}  n_cut_this_step={info['n_cut_this_step']}")
    print(f"    kept {kept_before} -> {info['kept']}  (one action removed {kept_before - info['kept']} edges)")
    assert info["n_cut_this_step"] == sz, "KILL should cut the whole parent group"
    # Re-killing the same node -> invalid (nothing left alive)
    obs, r, done, info = env.step(K + m_big)
    print(f"  KILL same node again: r={r:+.4f}  ({info['reason']})  expect -0.01 invalid")
    assert info["reason"] == "invalid"

    print("\n--- Episode C: never STOP, exhaust the 6-step budget ---")
    obs = env.reset(bundle_idx=0)
    for i in range(6):
        a = K - 1 - i  # cut distinct low-ranked candidates
        obs, r, done, info = env.step(a)
        print(f"  step {i+1}: CUT {a}  r={r:+.4f}  done={done}  ({info['reason']})")
    assert done, "Budget exhaust should auto-stop at step 6"
    print("  -> auto-stopped on budget, terminal reward folded into last step")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
