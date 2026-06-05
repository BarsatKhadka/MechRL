"""Unit test for BatchCutPolicy — no GPT-2, just the policy math.

Critical checks:
  1. act() and evaluate() return the SAME log-prob for the same action
     (if not, PPO's importance ratio is garbage).
  2. gradients flow to the edge head AND the size head.
  3. autoregressive picks are distinct and were alive when chosen.
  4. the STOP path works.
"""

from __future__ import annotations

import torch

from mechrl.env import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, N_GLOBALS
from mechrl.agent.batch_policy import BatchCutPolicy


def fake_obs(K=60, M=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "edge_features": torch.randn(K, EDGE_FEATURE_DIM, generator=g),
        "node_features": torch.randn(M, NODE_FEATURE_DIM, generator=g),
        "edge_alive": torch.ones(K),
        "kill_alive_frac": torch.rand(M, generator=g),
        "globals": torch.randn(N_GLOBALS, generator=g),
    }


def test_consistency_and_grads():
    torch.manual_seed(0)
    policy = BatchCutPolicy(batch_sizes=(1, 3, 10, 30))
    obs = fake_obs()

    # Force a BATCH action (re-sample until we don't get STOP) so we exercise
    # the autoregressive path.
    for attempt in range(50):
        torch.manual_seed(attempt + 1)
        a, logp_act, ent_act, val_act = policy.act(obs)
        if a["type"] == "batch" and len(a["edges"]) >= 2:
            break
    assert a["type"] == "batch", "never sampled a batch action"
    print(f"sampled batch: size_idx={a['size_idx']} N={len(a['edges'])} edges[:5]={a['edges'][:5]}")

    # (3) picks distinct + were alive (all alive in this obs)
    assert len(set(a["edges"])) == len(a["edges"]), "duplicate picks!"
    assert all(0 <= i < obs["edge_alive"].numel() for i in a["edges"]), "pick out of range"

    # (1) consistency: evaluate the SAME action -> same log-prob
    logp_eval, ent_eval, val_eval = policy.evaluate(obs, a)
    diff = abs(float(logp_act) - float(logp_eval))
    print(f"logp act={float(logp_act):.6f}  eval={float(logp_eval):.6f}  |diff|={diff:.2e}")
    assert diff < 1e-4, f"act/evaluate log-prob mismatch: {diff}"

    # value should match too (same forward on same obs)
    assert abs(float(val_act) - float(val_eval)) < 1e-5

    # (2) gradients flow to edge head AND size head
    loss = -(logp_eval) + val_eval.pow(2) - 0.01 * ent_eval
    policy.zero_grad()
    loss.backward()
    eh = policy.edge_head[0].weight.grad
    sh = policy.size_head[0].weight.grad
    assert eh is not None and eh.abs().sum() > 0, "edge head got no gradient"
    assert sh is not None and sh.abs().sum() > 0, "size head got no gradient"
    print(f"edge_head grad norm {eh.norm():.4f}  size_head grad norm {sh.norm():.4f}")

    # (4) STOP path
    # build a logits regime that forces stop by checking evaluate on a stop action
    logp_s, ent_s, val_s = policy.evaluate(obs, {"type": "stop"})
    assert torch.isfinite(logp_s) and torch.isfinite(ent_s)
    print(f"stop: logp={float(logp_s):.4f} entropy={float(ent_s):.4f}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    test_consistency_and_grads()
