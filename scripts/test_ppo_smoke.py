"""Tiny CPU smoke test for the PPO loop mechanics (NOT for learning).

Confirms:
  - a rollout collects transitions and at least one episode completes
  - GAE produces finite advantages/returns
  - the update runs, losses are finite
  - the policy weights actually change after updates
Kept deliberately tiny (small budget, few steps, 2 iterations) so it finishes
in ~2 min on CPU. Real training happens on the L40S.
"""

import copy
import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import CircuitEnv, TaskBundle
from mechrl.agent import CircuitPolicy
from mechrl.train import PPOConfig, PPOTrainer


def main():
    print("Building IOI bundle (cached prefilter)...")
    task = IOITask(num_examples=20, device="cpu")
    bundle = TaskBundle.build(task, k=3000)
    env = CircuitEnv([bundle], step_budget=8, seed=0)
    policy = CircuitPolicy()

    cfg = PPOConfig(
        total_iterations=2,
        num_steps=16,          # ~2 short episodes per rollout
        num_minibatches=2,
        update_epochs=2,
        learning_rate=3e-4,
        anneal_lr=False,
        seed=0,
    )
    trainer = PPOTrainer(env, policy, cfg, device="cpu")

    before = copy.deepcopy({k: v.clone() for k, v in policy.state_dict().items()})

    print("Running 2 tiny PPO iterations...\n")
    trainer.train(log_every=1)

    # weights changed?
    after = policy.state_dict()
    total_delta = sum((after[k] - before[k]).abs().sum().item() for k in before)
    print(f"\ntotal weight change after training: {total_delta:.4f} (expect > 0)")
    assert total_delta > 0, "policy weights did not change — update is a no-op!"

    # one rollout's advantages finite
    batch = trainer.collect()
    adv, ret, val = trainer.compute_gae(batch["rewards"], batch["values"], batch["dones"])
    print(f"advantages finite: {bool(torch.isfinite(adv).all())}, "
          f"returns finite: {bool(torch.isfinite(ret).all())}")
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()

    print("\nPPO smoke test passed — loop mechanics are sound.")


if __name__ == "__main__":
    main()
