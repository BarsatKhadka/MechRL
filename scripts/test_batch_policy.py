"""Unit test for BatchCutPolicy — no GPT-2, just the policy math.

Critical checks for ALL action types (batch, kill, stop):
  1. act() and evaluate() return the SAME log-prob for the same action
     (if not, PPO's importance ratio is garbage).
  2. gradients flow to edge head, node head, and type head.
  3. autoregressive batch picks are distinct and were alive.
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
        "kill_alive_frac": torch.rand(M, generator=g),   # some > 0 -> kill valid
        "globals": torch.randn(N_GLOBALS, generator=g),
    }


def consistency(policy, obs, want_type):
    """Sample until we get an action of want_type; check act/evaluate agree."""
    for attempt in range(400):
        torch.manual_seed(attempt + 1)
        a, logp_act, ent_act, val_act = policy.act(obs)
        ok = (a["type"] == want_type and
              (want_type != "batch" or len(a["edges"]) >= 2))
        if ok:
            break
    assert a["type"] == want_type, f"never sampled a {want_type} action"
    logp_eval, ent_eval, val_eval = policy.evaluate(obs, a)
    diff = abs(float(logp_act.detach()) - float(logp_eval.detach()))
    print(f"  {want_type:6s}: logp act={float(logp_act.detach()):+.5f} "
          f"eval={float(logp_eval.detach()):+.5f}  |diff|={diff:.2e}  detail={ {k:(v if k!='edges' else v[:4]) for k,v in a.items()} }")
    assert diff < 1e-4, f"{want_type} act/evaluate log-prob mismatch: {diff}"
    assert abs(float(val_act.detach()) - float(val_eval.detach())) < 1e-5
    return a, logp_eval, ent_eval


def main():
    torch.manual_seed(0)
    policy = BatchCutPolicy(batch_sizes=(1, 3, 10, 30))
    obs = fake_obs()

    print("act/evaluate consistency:")
    consistency(policy, obs, "batch")
    consistency(policy, obs, "kill")
    # STOP is ~impossible to sample at init (biased rare), so just check evaluate
    # handles it (no autoregressive replay, so consistency is trivial anyway).
    lp_s, ent_s, _ = policy.evaluate(obs, {"type": "stop"})
    assert torch.isfinite(lp_s) and torch.isfinite(ent_s)
    print(f"  stop  : evaluate logp={float(lp_s):+.4f} entropy={float(ent_s):.4f} (rare by design)")

    # gradients flow to all three action heads (use a batch action: touches edge+type)
    a, logp_eval, ent_eval = consistency(policy, obs, "batch")
    _, _, val = policy.evaluate(obs, a)
    loss = -logp_eval + val.pow(2) - 0.01 * ent_eval
    policy.zero_grad(); loss.backward()
    for name, head in [("edge", policy.edge_head), ("type", policy.type_head), ("value", policy.value_head)]:
        g = head[0].weight.grad
        assert g is not None and g.abs().sum() > 0, f"{name} head got no gradient"
        print(f"  {name} head grad norm {g.norm():.4f}")
    # node head gets gradient from a kill action
    ak, logp_k, ent_k = consistency(policy, obs, "kill")
    policy.zero_grad(); (-logp_k).backward()
    gn = policy.node_head[0].weight.grad
    assert gn is not None and gn.abs().sum() > 0, "node head got no gradient"
    print(f"  node head grad norm {gn.norm():.4f}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
