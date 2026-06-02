"""CircuitPolicy — actor-critic for the circuit-finding agent.

Feature-scoring (pointer-style) policy: one shared MLP scores every edge, one
scores every node, plus a STOP head and a value (critic) head. Parameter count
is INDEPENDENT of the number of candidates K — the K/M show up only as a batch
dimension. Already-cut actions are masked to a large negative logit so the agent
can never pick an invalid move.

Forward pass (single observation):
  1. embed each edge (its features + alive bit) and each node (features + alive frac)
  2. pool edge & node embeddings (mean + max) and combine with globals -> context
  3. score each edge from (its embedding + context); same for nodes; plus STOP
  4. mask dead actions; value = value_head(context)

Designed to consume the observation dict produced by CircuitEnv directly:
  edge_features [K,16], node_features [M,7], edge_alive [K],
  kill_alive_frac [M], globals [5].
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from mechrl.env import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, N_GLOBALS

MASK_VALUE = -1e9   # logit for invalid actions (softmax -> ~0 probability)


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class CircuitPolicy(nn.Module):
    def __init__(
        self,
        edge_dim: int = EDGE_FEATURE_DIM,
        node_dim: int = NODE_FEATURE_DIM,
        n_globals: int = N_GLOBALS,
        hidden: int = 128,
    ):
        super().__init__()
        self.hidden = hidden
        # +1 on each: edge gets its alive bit, node gets its alive-fraction.
        self.edge_embed = _mlp([edge_dim + 1, hidden, hidden])
        self.node_embed = _mlp([node_dim + 1, hidden, hidden])
        # context sees pooled (mean+max) edges, pooled (mean+max) nodes, and globals
        ctx_in = 2 * hidden + 2 * hidden + n_globals
        self.context = _mlp([ctx_in, hidden, hidden])
        # scoring heads: each item's embedding concatenated with the context
        self.edge_head = _mlp([hidden + hidden, hidden, 1])
        self.node_head = _mlp([hidden + hidden, hidden, 1])
        self.stop_head = _mlp([hidden, hidden, 1])
        self.value_head = _mlp([hidden, hidden, 1])

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [K+M+1], value scalar) for one observation."""
        ef = obs["edge_features"]                       # [K, edge_dim]
        nf = obs["node_features"]                       # [M, node_dim]
        ea = obs["edge_alive"].unsqueeze(-1)            # [K, 1]
        naf = obs["kill_alive_frac"].unsqueeze(-1)      # [M, 1]
        g = obs["globals"]                              # [n_globals]

        e = self.edge_embed(torch.cat([ef, ea], dim=-1))   # [K, H]
        n = self.node_embed(torch.cat([nf, naf], dim=-1))  # [M, H]

        pooled = torch.cat(
            [e.mean(0), e.amax(0), n.mean(0), n.amax(0), g], dim=-1
        )                                                  # [4H + n_globals]
        ctx = self.context(pooled)                         # [H]

        ctx_e = ctx.unsqueeze(0).expand(e.size(0), -1)     # [K, H]
        ctx_n = ctx.unsqueeze(0).expand(n.size(0), -1)     # [M, H]
        edge_logits = self.edge_head(torch.cat([e, ctx_e], dim=-1)).squeeze(-1)   # [K]
        node_logits = self.node_head(torch.cat([n, ctx_n], dim=-1)).squeeze(-1)   # [M]
        stop_logit = self.stop_head(ctx)                                          # [1]

        logits = torch.cat([edge_logits, node_logits, stop_logit], dim=0)         # [K+M+1]

        # ---- action masking ----
        edge_valid = obs["edge_alive"] > 0.5
        node_valid = obs["kill_alive_frac"] > 0.0
        stop_valid = torch.ones(1, dtype=torch.bool, device=logits.device)
        valid = torch.cat([edge_valid, node_valid, stop_valid], dim=0)
        logits = logits.masked_fill(~valid, MASK_VALUE)

        value = self.value_head(ctx).squeeze(-1)
        return logits, value

    # ---- PPO helpers ----

    def get_value(self, obs):
        """Critic only — used to bootstrap the value at the end of a rollout."""
        _, value = self.forward(obs)
        return value

    def act(self, obs):
        """Sample an action. Returns (action, log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), dist.log_prob(a), dist.entropy(), value

    def evaluate(self, obs, action: torch.Tensor):
        """Recompute (log_prob, entropy, value) for a stored (obs, action). Used in the PPO update."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value
