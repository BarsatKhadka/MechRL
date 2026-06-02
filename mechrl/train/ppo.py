"""PPO trainer for the circuit-finding agent.

Adapted from CleanRL's single-file PPO (https://docs.cleanrl.dev/rl-algorithms/ppo/).
The ALGORITHM is unchanged — GAE advantages, clipped surrogate objective, clipped
value loss, entropy bonus, global grad-norm clipping. What we changed is the
PLUMBING for our setting:

  - Observations are DICTS of variable-sized tensors (not a flat vector), so we
    store a python list of obs instead of a preallocated tensor.
  - The actor is feature-scored + masked with a variable action dim, so we never
    batch the policy forward across samples; we loop policy.evaluate() per stored
    step. That's cheap — the per-step cost was the GPT-2 forward during ROLLOUT,
    which is already paid; the policy MLP is negligible.
  - One env at a time (each env owns a GPT-2), so num_envs is effectively 1 and
    the env auto-reset is done manually on `done`.

Multi-task works for free: env.reset() may sample a different task (different K/M)
each episode — because we evaluate per-sample, mixed action dims never collide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    total_iterations: int = 50
    num_steps: int = 128          # env transitions collected per rollout
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    anneal_lr: bool = True
    target_kl: Optional[float] = None
    seed: int = 0


class PPOTrainer:
    def __init__(self, env, policy, cfg: PPOConfig, device: str = "cpu"):
        self.env = env
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self._next_obs = None
        self._next_done = False

    def _to_device(self, obs):
        return {k: v.to(self.device) for k, v in obs.items()}

    # ---- rollout ----

    def collect(self):
        """Run num_steps env transitions; manual reset on done (single env)."""
        cfg = self.cfg
        obs_buf, act_buf = [], []
        logp_buf, rew_buf, done_buf, val_buf = [], [], [], []
        ep_returns, ep_infos = [], []

        if self._next_obs is None:
            self._next_obs = self._to_device(self.env.reset())
            self._next_done = False

        ep_ret = 0.0
        for _ in range(cfg.num_steps):
            obs = self._next_obs
            with torch.no_grad():
                a, logp, _, val = self.policy.act(obs)

            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(float(logp))
            val_buf.append(float(val))
            done_buf.append(float(self._next_done))   # done flag for THIS obs (CleanRL convention)

            nobs, r, done, info = self.env.step(a)
            rew_buf.append(float(r))
            ep_ret += float(r)

            if done:
                ep_returns.append(ep_ret)
                ep_infos.append(info)
                ep_ret = 0.0
                nobs = self.env.reset()       # manual auto-reset
                self._next_done = True
            else:
                self._next_done = False
            self._next_obs = self._to_device(nobs)

        return dict(
            obs=obs_buf, actions=act_buf, logprobs=logp_buf, rewards=rew_buf,
            dones=done_buf, values=val_buf, ep_returns=ep_returns, ep_infos=ep_infos,
        )

    # ---- advantages (GAE, identical to CleanRL) ----

    def compute_gae(self, rewards, values, dones):
        cfg = self.cfg
        T = len(rewards)
        with torch.no_grad():
            next_value = float(self.policy.get_value(self._next_obs))
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        advantages = torch.zeros(T, device=self.device)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - float(self._next_done)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        return advantages, returns, values

    # ---- PPO update ----

    def update(self, batch, advantages, returns, values):
        cfg = self.cfg
        T = len(batch["obs"])
        b_logprobs = torch.tensor(batch["logprobs"], dtype=torch.float32, device=self.device)
        b_actions = batch["actions"]
        b_obs = batch["obs"]
        b_adv = advantages
        b_ret = returns
        b_val = values

        mb_size = max(1, T // cfg.num_minibatches)
        inds = np.arange(T)
        approx_kl = 0.0
        last = {}
        for _ in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, T, mb_size):
                mb = inds[start:start + mb_size]

                # recompute logp/entropy/value per stored step (variable action dim)
                newlogp, entropy, newval = [], [], []
                for i in mb:
                    lp, ent, v = self.policy.evaluate(
                        b_obs[i], torch.tensor(b_actions[i], device=self.device)
                    )
                    newlogp.append(lp); entropy.append(ent); newval.append(v)
                newlogp = torch.stack(newlogp)
                entropy = torch.stack(entropy)
                newval = torch.stack(newval)

                logratio = newlogp - b_logprobs[torch.as_tensor(mb)]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()

                mb_adv = b_adv[torch.as_tensor(mb)]
                if cfg.norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # clipped policy loss
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # value loss
                v_loss = 0.5 * ((newval - b_ret[torch.as_tensor(mb)]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                last = dict(pg_loss=pg_loss.item(), v_loss=v_loss.item(),
                            entropy=ent_loss.item(), approx_kl=approx_kl)

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        # explained variance (how well the critic predicts returns)
        y_pred = b_val.cpu().numpy()
        y_true = b_ret.cpu().numpy()
        var_y = np.var(y_true)
        last["explained_var"] = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
        return last

    # ---- main loop ----

    def train(self, log_every: int = 1, save_dir=None, save_every: int = 0,
              metrics_path=None) -> List[dict]:
        cfg = self.cfg
        history: List[dict] = []
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        mf = open(metrics_path, "a") if metrics_path is not None else None

        for it in range(1, cfg.total_iterations + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (it - 1.0) / cfg.total_iterations
                self.opt.param_groups[0]["lr"] = frac * cfg.learning_rate

            batch = self.collect()
            adv, ret, val = self.compute_gae(batch["rewards"], batch["values"], batch["dones"])
            stats = self.update(batch, adv, ret, val)

            eps = batch["ep_returns"]
            faiths = [i.get("faith", float("nan")) for i in batch["ep_infos"]]
            kepts = [i.get("kept", -1) for i in batch["ep_infos"]]
            rec = {
                "iter": it,
                "episodes": len(eps),
                "return": float(np.mean(eps)) if eps else float("nan"),
                "faith": float(np.mean(faiths)) if faiths else float("nan"),
                "kept": float(np.mean(kepts)) if kepts else float("nan"),
                "lr": self.opt.param_groups[0]["lr"],
                **stats,
            }
            history.append(rec)

            if it % log_every == 0:
                print(
                    f"iter {it:4d} | episodes {rec['episodes']:2d} | "
                    f"return {rec['return']:+.4f} | faith {rec['faith']:.3f} | kept {rec['kept']:7.1f} | "
                    f"pg {stats['pg_loss']:+.4f} | v {stats['v_loss']:.4f} | "
                    f"ent {stats['entropy']:.3f} | kl {stats['approx_kl']:.4f} | "
                    f"expl_var {stats['explained_var']:+.2f}",
                    flush=True,
                )
            if mf is not None:
                mf.write(json.dumps(rec, default=float) + "\n")   # default=float: tolerate np scalars
                mf.flush()
            if save_dir is not None and save_every and it % save_every == 0:
                torch.save(self.policy.state_dict(), Path(save_dir) / f"policy_iter{it}.pt")

        if save_dir is not None:
            torch.save(self.policy.state_dict(), Path(save_dir) / "policy_final.pt")
        if mf is not None:
            mf.close()
        return history
