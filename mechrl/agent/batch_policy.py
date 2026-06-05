"""BatchCutPolicy — actor-critic with a learned, autoregressive batch-cut action.

Replaces the flat (single-cut / kill / stop) action space with:

    action = STOP                       end the episode
           | BATCH(size_idx, [edges])   cut these edges (chosen autoregressively)

The agent makes two coupled decisions per step (factored action):
  1. SIZE head -> pick {STOP, N=batch_sizes[0], N=batch_sizes[1], ...}
  2. EDGE head -> if not STOP, sample N edges ONE AT A TIME, re-scoring between
     picks (autoregressive). Marking a pick "cut" flips its alive-bit, which
     shifts the context the edge head reads -> the next pick conditions on it.
     This is what lets the agent learn "don't cut B right after A" (OR-gate /
     backup-head coordination). See reward-loop / design notes.

Why autoregressive solves both problems:
  - Problem 1 (staleness): picks see each other, so redundant pairs can be split.
  - Problem 2 (trainability): each pick is SAMPLED, so the whole batch has a
    real differentiable log-prob (sum of per-pick + the size choice). A hard
    top-N sort would be deterministic -> zero gradient -> ranking never learns.

Cost note: the N re-scorings are cheap POLICY forwards (no GPT-2). The single
expensive GPT-2 faith eval happens once per step, in the env, after the batch.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from mechrl.env import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, N_GLOBALS

MASK_VALUE = -1e9
ALIVE_GLOBAL_IDX = 2   # globals[2] = alive fraction (see CircuitEnv._observation)


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class BatchCutPolicy(nn.Module):
    def __init__(
        self,
        edge_dim: int = EDGE_FEATURE_DIM,
        node_dim: int = NODE_FEATURE_DIM,
        n_globals: int = N_GLOBALS,
        hidden: int = 128,
        batch_sizes: Tuple[int, ...] = (1, 3, 10, 30, 100),
    ):
        super().__init__()
        self.hidden = hidden
        self.batch_sizes = list(batch_sizes)
        self.edge_embed = _mlp([edge_dim + 1, hidden, hidden])
        self.node_embed = _mlp([node_dim + 1, hidden, hidden])
        ctx_in = 2 * hidden + 2 * hidden + n_globals
        self.context = _mlp([ctx_in, hidden, hidden])
        self.edge_head = _mlp([hidden + hidden, hidden, 1])
        # size head: index 0 = STOP, indices 1..B = batch_sizes[0..B-1]
        self.size_head = _mlp([hidden, hidden, 1 + len(self.batch_sizes)])
        self.value_head = _mlp([hidden, hidden, 1])

    # ---- shared trunk ----

    def _context_and_edge_logits(self, obs: Dict[str, torch.Tensor]):
        ef = obs["edge_features"]
        nf = obs["node_features"]
        ea = obs["edge_alive"].unsqueeze(-1)
        naf = obs["kill_alive_frac"].unsqueeze(-1)
        g = obs["globals"]

        e = self.edge_embed(torch.cat([ef, ea], dim=-1))   # [K, H]
        n = self.node_embed(torch.cat([nf, naf], dim=-1))  # [M, H]
        pooled = torch.cat([e.mean(0), e.amax(0), n.mean(0), n.amax(0), g], dim=-1)
        ctx = self.context(pooled)                         # [H]

        ctx_e = ctx.unsqueeze(0).expand(e.size(0), -1)
        edge_logits = self.edge_head(torch.cat([e, ctx_e], dim=-1)).squeeze(-1)  # [K]
        alive = obs["edge_alive"] > 0.5
        edge_logits = edge_logits.masked_fill(~alive, MASK_VALUE)
        return ctx, edge_logits

    def forward(self, obs):
        ctx, edge_logits = self._context_and_edge_logits(obs)
        size_logits = self.size_head(ctx)                  # [1 + B]
        value = self.value_head(ctx).squeeze(-1)
        return edge_logits, size_logits, value

    def get_value(self, obs):
        return self.forward(obs)[2]

    # ---- obs simulation during autoregressive sampling (NO GPT-2) ----

    def _clone_dynamic(self, obs):
        """Copy only the tensors we mutate while sampling a batch."""
        o = dict(obs)
        o["edge_alive"] = obs["edge_alive"].clone()
        o["globals"] = obs["globals"].clone()
        return o

    def _mark_cut(self, obs, idx: int):
        """Flip an edge to cut and refresh the alive-fraction global (what the
        re-score reads). Faith globals are deliberately NOT updated — within a
        batch we predict from the cut pattern, we don't re-measure faith."""
        obs["edge_alive"][idx] = 0.0
        obs["globals"][ALIVE_GLOBAL_IDX] = obs["edge_alive"].mean()

    # ---- acting (rollout) ----

    def act(self, obs, greedy: bool = False):
        """Sample (or, if greedy, argmax) a composite action.
        Returns (action_dict, logp, entropy, value).

        action_dict is one of:
          {"type": "stop"}
          {"type": "batch", "size_idx": int, "edges": [candidate-local idxs]}

        greedy=True (deterministic) is used to EXTRACT the circuit the policy
        commits to — argmax the size and each edge pick instead of sampling.
        """
        edge_logits, size_logits, value = self.forward(obs)
        size_dist = Categorical(logits=size_logits)
        sc = size_logits.argmax() if greedy else size_dist.sample()
        logp = size_dist.log_prob(sc)
        ent_size = size_dist.entropy()
        sci = int(sc.item())

        if sci == 0:
            return {"type": "stop"}, logp, ent_size, value

        n_target = self.batch_sizes[sci - 1]
        work = self._clone_dynamic(obs)
        edges: List[int] = []
        edge_ents: List[torch.Tensor] = []
        for _ in range(n_target):
            _, el = self._context_and_edge_logits(work)
            if (el > MASK_VALUE / 2).sum() == 0:           # no alive edges left
                break
            d = Categorical(logits=el)
            pick = el.argmax() if greedy else d.sample()
            logp = logp + d.log_prob(pick)                 # logp: full SUM (joint log-prob)
            edge_ents.append(d.entropy())
            pi = int(pick.item())
            edges.append(pi)
            self._mark_cut(work, pi)

        # entropy bonus = size entropy + MEAN per-pick entropy. Averaging (not
        # summing) keeps the bonus scale-invariant to batch size, so ent_coef
        # stays comparable to the flat policy. logp stays the full sum.
        entropy = ent_size + (torch.stack(edge_ents).mean() if edge_ents else 0.0 * ent_size)
        return {"type": "batch", "size_idx": sci, "edges": edges}, logp, entropy, value

    # ---- evaluating a stored action (PPO update) ----

    def evaluate(self, obs, action):
        """Recompute (logp, entropy, value) for a STORED action via teacher
        forcing — replay the same size choice and the same edge sequence so the
        log-prob matches what act() produced (modulo weight updates)."""
        edge_logits, size_logits, value = self.forward(obs)
        size_dist = Categorical(logits=size_logits)

        if action["type"] == "stop":
            sc = torch.zeros((), dtype=torch.long, device=size_logits.device)
            return size_dist.log_prob(sc), size_dist.entropy(), value

        sci = action["size_idx"]
        sc = torch.as_tensor(sci, dtype=torch.long, device=size_logits.device)
        logp = size_dist.log_prob(sc)
        ent_size = size_dist.entropy()

        work = self._clone_dynamic(obs)
        edge_ents: List[torch.Tensor] = []
        for pi in action["edges"]:
            _, el = self._context_and_edge_logits(work)
            d = Categorical(logits=el)
            t = torch.as_tensor(pi, dtype=torch.long, device=el.device)
            logp = logp + d.log_prob(t)                    # logp: full SUM
            edge_ents.append(d.entropy())
            self._mark_cut(work, pi)
        entropy = ent_size + (torch.stack(edge_ents).mean() if edge_ents else 0.0 * ent_size)
        return logp, entropy, value
