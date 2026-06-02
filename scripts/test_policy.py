"""Fast unit test for CircuitPolicy on a SYNTHETIC observation (no GPT-2 load).

Checks the wiring:
  - output shapes: logits = K+M+1, value = scalar
  - parameter count is independent of K (score 3000 vs 6000 edges, same #params)
  - action masking: cut/killed actions get ~0 probability, STOP always valid
  - distribution is valid (sums to 1) and never samples a masked action
  - gradients flow through actor and critic
"""

import torch
from torch.distributions import Categorical

from mechrl.agent import CircuitPolicy
from mechrl.agent.policy import MASK_VALUE
from mechrl.env import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, N_GLOBALS


def fake_obs(K=3000, M=150):
    return {
        "edge_features": torch.randn(K, EDGE_FEATURE_DIM),
        "node_features": torch.randn(M, NODE_FEATURE_DIM),
        "edge_alive": torch.ones(K),
        "kill_alive_frac": torch.ones(M),
        "globals": torch.randn(N_GLOBALS),
    }


def main():
    torch.manual_seed(0)
    policy = CircuitPolicy(hidden=128)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy parameters: {n_params:,}")

    K, M = 3000, 150
    obs = fake_obs(K, M)
    logits, value = policy(obs)
    print(f"\n[1] shapes: logits={tuple(logits.shape)} (expect {K+M+1}), value={tuple(value.shape)} (scalar)")
    assert logits.shape == (K + M + 1,)
    assert value.dim() == 0

    # [2] param count independent of K
    obs_big = fake_obs(K=6000, M=300)
    logits_big, _ = policy(obs_big)
    n_params2 = sum(p.numel() for p in policy.parameters())
    print(f"[2] K=6000 -> logits={tuple(logits_big.shape)}; params still {n_params2:,} (same network)")
    assert n_params == n_params2
    assert logits_big.shape == (6000 + 300 + 1,)

    # [3] masking: kill first 100 edges and first 10 nodes
    obs_m = fake_obs(K, M)
    obs_m["edge_alive"][:100] = 0.0
    obs_m["kill_alive_frac"][:10] = 0.0
    logits_m, _ = policy(obs_m)
    print(f"[3] masked 100 dead edges -> max logit among them: {logits_m[:100].max():.0f} (expect {MASK_VALUE:.0f})")
    assert (logits_m[:100] <= MASK_VALUE + 1).all(), "dead edges not masked"
    assert (logits_m[K:K + 10] <= MASK_VALUE + 1).all(), "dead nodes not masked"
    assert logits_m[-1] > MASK_VALUE + 1, "STOP should always be valid"

    # [4] distribution valid; masked actions get ~0 probability
    dist = Categorical(logits=logits_m)
    probs = dist.probs
    print(f"[4] prob sum={probs.sum():.4f} (expect 1.0); prob mass on 110 masked actions="
          f"{probs[:100].sum() + probs[K:K+10].sum():.2e} (expect ~0)")
    assert abs(probs.sum().item() - 1.0) < 1e-4
    assert (probs[:100].sum() + probs[K:K + 10].sum()).item() < 1e-6

    # [5] sampling never returns a masked action (1000 draws)
    bad = 0
    for _ in range(1000):
        a, _, _, _ = policy.act(obs_m)
        if a < 100 or (K <= a < K + 10):
            bad += 1
    print(f"[5] masked actions sampled in 1000 draws: {bad} (expect 0)")
    assert bad == 0

    # [6] gradients flow (a fake PPO-ish loss)
    a, logp, ent, val = policy.act(obs)
    loss = -(logp * 1.0) - 0.01 * ent + (val - 0.5).pow(2)
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in policy.parameters()]
    print(f"[6] params receiving gradient: {sum(grads)}/{len(grads)}")
    assert all(grads), "some parameters got no gradient"

    print("\nPolicy wiring checks passed.")


if __name__ == "__main__":
    main()
