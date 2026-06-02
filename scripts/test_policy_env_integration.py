"""Integration test: CircuitPolicy on a REAL CircuitEnv observation.

Closes the seam between Layer 1 (env) and Layer 2 (policy):
  - policy consumes env.reset() obs without shape/key errors
  - logits width matches env.action_dim (changes as K/M are real)
  - policy.act picks a VALID action (never a masked one), env accepts it
  - works across several steps as the alive-state changes
No training — just confirms the two halves connect.
"""

import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import CircuitEnv, TaskBundle
from mechrl.agent import CircuitPolicy


def main():
    print("Building IOI bundle (cached prefilter)...")
    task = IOITask(num_examples=20, device="cpu")
    bundle = TaskBundle.build(task, k=3000)
    env = CircuitEnv([bundle], step_budget=8, seed=0)
    policy = CircuitPolicy()

    obs = env.reset(bundle_idx=0)
    print(f"env.action_dim = {env.action_dim}  (K={env.n_candidates} + M={env.n_kill} + STOP)")

    logits, value = policy(obs)
    print(f"\n[1] policy(real obs): logits={tuple(logits.shape)}, value={value.item():+.3f}")
    assert logits.shape == (env.action_dim,), "logits width != action_dim"

    print("\n[2] drive a short episode with policy.act():")
    K, M = env.n_candidates, env.n_kill
    done = False
    steps = 0
    while not done and steps < 8:
        a, logp, ent, val = policy.act(obs)
        kind = "CUT" if a < K else ("KILL" if a < K + M else "STOP")
        obs, r, done, info = env.step(a)
        # an untrained policy should still never produce an INVALID action (masking)
        assert info["reason"] != "invalid", f"policy picked a masked action {a}!"
        print(f"  step {steps+1}: action={a:4d} [{kind}] logp={logp.item():+.2f} "
              f"value={val.item():+.2f} r={r:+.4f} faith={info['faith']:.3f} ({info['reason']})")
        steps += 1

    print("\n[3] evaluate() recomputes logp/value for a stored action:")
    obs = env.reset(bundle_idx=0)
    a, logp_act, _, _ = policy.act(obs)
    logp_eval, ent_eval, val_eval = policy.evaluate(obs, torch.tensor(a))
    print(f"  act logp={logp_act.item():+.4f}  evaluate logp={logp_eval.item():+.4f} (should match)")
    assert torch.allclose(logp_act, logp_eval, atol=1e-5), "act/evaluate disagree"

    print("\nIntegration test passed — policy and env connect end-to-end.")


if __name__ == "__main__":
    main()
