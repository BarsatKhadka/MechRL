"""TransformerCutPolicy — same actor-critic as BatchCutPolicy, but edges run
through a SELF-ATTENTION encoder so each edge's representation depends on the
whole subgraph (not just its own features).

Motivation (vs BatchCutPolicy's MLP):
  - BatchCutPolicy embeds each edge INDEPENDENTLY and mixes them only via a
    mean/max pool -> it scores edge i from (edge i's own features, a blurry global
    summary). It can't see that two edges are redundant, or that edge C only
    matters if edge D is kept. So it keeps the right NUMBER of edges but not the
    optimal SET (the IOI selection tax: ~1100 edges @ 0.87 vs single-task 0.96).
  - Here each edge is a TOKEN; a TransformerEncoder lets every edge attend to
    every other edge, so its representation encodes its ROLE in the circuit. The
    edge head then scores structure-aware reps -> can select the load-bearing set.

Design for tractability (keeps the autoregressive selective-cut machinery intact):
  - The encoder runs ONCE per env step (in _embed) over the alive edges -> attended
    reps e [K,H]. The autoregressive batch loop then re-pools the alive context and
    re-scores from those FIXED reps (no re-attention per pick) -- exactly mirroring
    BatchCutPolicy's loop, which also embeds once then re-pools. Cutting <=30 edges
    out of thousands barely changes the structure, so not re-attending per pick is a
    fine approximation and keeps cost ~O(K^2) per step (not O(K^3) per batch).
  - norm_first=True (pre-LN) for RL stability (cf. Parisotto 2020, "Stabilizing
    Transformers for RL").

Interface is identical to BatchCutPolicy: act / evaluate / get_value / forward,
same action dicts (stop | batch | kill) and (logp, entropy, value) returns -- so
it drops into PPOTrainer / train_agent via --policy transformer.
"""

from __future__ import annotations

from typing import List, Tuple

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


class TransformerCutPolicy(nn.Module):
    def __init__(
        self,
        edge_dim: int = EDGE_FEATURE_DIM,
        node_dim: int = NODE_FEATURE_DIM,
        n_globals: int = N_GLOBALS,
        hidden: int = 128,
        batch_sizes: Tuple[int, ...] = (1, 3, 10, 30, 100),
        n_layers: int = 2,
        n_heads: int = 4,
        ff: int = 256,
    ):
        super().__init__()
        self.hidden = hidden
        self.batch_sizes = list(batch_sizes)
        self.n_sizes = len(self.batch_sizes)
        self.kill_idx = 1 + self.n_sizes
        self.n_types = 2 + self.n_sizes

        # edge tokens -> attention encoder (the only structural change vs BatchCutPolicy)
        self.edge_in = nn.Linear(edge_dim + 1, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ff,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # nodes stay simple (KILL is the secondary action); independent embed
        self.node_embed = _mlp([node_dim + 1, hidden, hidden])

        ctx_in = 2 * hidden + 2 * hidden + n_globals
        self.context = _mlp([ctx_in, hidden, hidden])
        self.edge_head = _mlp([hidden + hidden, hidden, 1])
        self.node_head = _mlp([hidden + hidden, hidden, 1])
        self.type_head = _mlp([hidden, hidden, self.n_types])
        self.value_head = _mlp([hidden, hidden, 1])

        # Same warm start as BatchCutPolicy: single-cut dominant, STOP rare at init.
        with torch.no_grad():
            self.type_head[-1].bias.zero_()
            self.type_head[-1].bias[0] = -5.0
            self.type_head[-1].bias[1] = 2.0

    # ---- embedding (the attention part) ----

    def _encode_edges(self, obs):
        """Run the self-attention encoder over edge tokens, masking dead edges out
        of the attention keys. Returns structure-aware reps e [K, H]."""
        ef = obs["edge_features"]
        ea = obs["edge_alive"].unsqueeze(-1)
        tok = self.edge_in(torch.cat([ef, ea], dim=-1)).unsqueeze(0)   # [1, K, H]
        dead = ~(obs["edge_alive"] > 0.5)                              # [K] True = dead
        if bool(dead.all()):
            return tok.squeeze(0)                                      # all dead -> skip (avoids NaN)
        e = self.encoder(tok, src_key_padding_mask=dead.unsqueeze(0))  # [1, K, H]
        return e.squeeze(0)

    def _ctx_from(self, e, n, g, edge_alive):
        """Context from attended edges (pooled over ALIVE edges) + nodes + globals.
        Re-callable in the autoregressive loop with an updated alive mask, reusing
        the fixed attended reps e (no re-attention)."""
        alive = edge_alive > 0.5
        if bool(alive.any()):
            ea = e[alive]
            e_mean, e_max = ea.mean(0), ea.amax(0)
        else:
            e_mean = e_max = torch.zeros(self.hidden, device=e.device, dtype=e.dtype)
        pooled = torch.cat([e_mean, e_max, n.mean(0), n.amax(0), g], dim=-1)
        return self.context(pooled)

    def _embed(self, obs):
        e = self._encode_edges(obs)                                    # [K, H] attended
        nf = obs["node_features"]
        naf = obs["kill_alive_frac"].unsqueeze(-1)
        n = self.node_embed(torch.cat([nf, naf], dim=-1))              # [M, H]
        ctx = self._ctx_from(e, n, obs["globals"], obs["edge_alive"])
        return ctx, e, n

    # ---- heads (identical to BatchCutPolicy) ----

    def _edge_logits(self, obs, ctx, e):
        ctx_e = ctx.unsqueeze(0).expand(e.size(0), -1)
        el = self.edge_head(torch.cat([e, ctx_e], dim=-1)).squeeze(-1)
        return el.masked_fill(~(obs["edge_alive"] > 0.5), MASK_VALUE)

    def _node_logits(self, obs, ctx, n):
        ctx_n = ctx.unsqueeze(0).expand(n.size(0), -1)
        nl = self.node_head(torch.cat([n, ctx_n], dim=-1)).squeeze(-1)
        return nl.masked_fill(~(obs["kill_alive_frac"] > 0.0), MASK_VALUE)

    def _type_logits(self, obs, ctx):
        tl = self.type_head(ctx)
        mask = torch.ones_like(tl, dtype=torch.bool)
        if not bool((obs["edge_alive"] > 0.5).any()):
            mask[1:1 + self.n_sizes] = False
        if not bool((obs["kill_alive_frac"] > 0.0).any()):
            mask[self.kill_idx] = False
        return tl.masked_fill(~mask, MASK_VALUE)

    def forward(self, obs):
        ctx, e, n = self._embed(obs)
        return (self._edge_logits(obs, ctx, e),
                self._node_logits(obs, ctx, n),
                self._type_logits(obs, ctx),
                self.value_head(ctx).squeeze(-1))

    def get_value(self, obs):
        return self.forward(obs)[3]

    # ---- obs simulation during autoregressive batch sampling (no GPT-2) ----

    def _clone_dynamic(self, obs):
        o = dict(obs)
        o["edge_alive"] = obs["edge_alive"].clone()
        o["globals"] = obs["globals"].clone()
        return o

    def _mark_cut(self, obs, idx: int):
        obs["edge_alive"][idx] = 0.0
        obs["globals"][ALIVE_GLOBAL_IDX] = obs["edge_alive"].mean()

    # ---- acting ----

    def act(self, obs, greedy: bool = False):
        ctx, e, n = self._embed(obs)
        type_logits = self._type_logits(obs, ctx)
        node_logits = self._node_logits(obs, ctx, n)
        value = self.value_head(ctx).squeeze(-1)

        type_dist = Categorical(logits=type_logits)
        ti = type_logits.argmax() if greedy else type_dist.sample()
        logp = type_dist.log_prob(ti)
        entropy = type_dist.entropy()
        t = int(ti.item())

        if t == 0:
            return {"type": "stop"}, logp, entropy, value

        if t == self.kill_idx:
            node_dist = Categorical(logits=node_logits)
            node = node_logits.argmax() if greedy else node_dist.sample()
            logp = logp + node_dist.log_prob(node)
            entropy = entropy + node_dist.entropy()
            return {"type": "kill", "node": int(node.item())}, logp, entropy, value

        # BATCH: autoregressively sample N edges, re-pooling context between picks
        # from the FIXED attended reps e (no re-attention -- see module docstring).
        n_target = self.batch_sizes[t - 1]
        work = self._clone_dynamic(obs)
        edges: List[int] = []
        edge_ents: List[torch.Tensor] = []
        for _ in range(n_target):
            cw = self._ctx_from(e, n, work["globals"], work["edge_alive"])
            el = self._edge_logits(work, cw, e)
            if (el > MASK_VALUE / 2).sum() == 0:
                break
            d = Categorical(logits=el)
            pick = el.argmax() if greedy else d.sample()
            logp = logp + d.log_prob(pick)
            edge_ents.append(d.entropy())
            pi = int(pick.item())
            edges.append(pi)
            self._mark_cut(work, pi)

        entropy = entropy + (torch.stack(edge_ents).mean() if edge_ents else 0.0 * entropy)
        return {"type": "batch", "size_idx": t, "edges": edges}, logp, entropy, value

    # ---- evaluating a stored action (PPO update, teacher-forced) ----

    def evaluate(self, obs, action):
        ctx, e, n = self._embed(obs)
        type_logits = self._type_logits(obs, ctx)
        node_logits = self._node_logits(obs, ctx, n)
        value = self.value_head(ctx).squeeze(-1)
        type_dist = Categorical(logits=type_logits)

        if action["type"] == "stop":
            sc = torch.zeros((), dtype=torch.long, device=type_logits.device)
            return type_dist.log_prob(sc), type_dist.entropy(), value

        if action["type"] == "kill":
            sc = torch.as_tensor(self.kill_idx, dtype=torch.long, device=type_logits.device)
            logp = type_dist.log_prob(sc)
            entropy = type_dist.entropy()
            node_dist = Categorical(logits=node_logits)
            nt = torch.as_tensor(action["node"], dtype=torch.long, device=node_logits.device)
            return logp + node_dist.log_prob(nt), entropy + node_dist.entropy(), value

        # BATCH -- teacher-forced replay of the stored edges (same loop as act()).
        sc = torch.as_tensor(action["size_idx"], dtype=torch.long, device=type_logits.device)
        logp = type_dist.log_prob(sc)
        ent_type = type_dist.entropy()
        work = self._clone_dynamic(obs)
        edge_ents: List[torch.Tensor] = []
        for pi in action["edges"]:
            cw = self._ctx_from(e, n, work["globals"], work["edge_alive"])
            el = self._edge_logits(work, cw, e)
            d = Categorical(logits=el)
            tt = torch.as_tensor(pi, dtype=torch.long, device=el.device)
            logp = logp + d.log_prob(tt)
            edge_ents.append(d.entropy())
            self._mark_cut(work, pi)
        entropy = ent_type + (torch.stack(edge_ents).mean() if edge_ents else 0.0 * ent_type)
        return logp, entropy, value
