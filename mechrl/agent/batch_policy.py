"""BatchCutPolicy — actor-critic with batch-cut AND kill-parent actions.

Per step the agent chooses one of:

    action = STOP                         end the episode
           | BATCH(size_idx, [edges])     cut N edges, chosen AUTOREGRESSIVELY
           | KILL(node)                   cut ALL of one parent node's alive edges

- BATCH is the selective bulk cut (Problem 1: autoregressive re-scoring between
  picks dodges OR-gate pairs; Problem 2: sampled -> differentiable joint log-prob).
- KILL is the blunt whole-node cut, kept because it's the workhorse that drove the
  flat agent down fast (great EARLY for clearly-dead nodes; the agent learns to
  stop using it once nodes have mixed important/redundant edges).
- STOP, the batch sizes, and KILL are one factored "type" head; BATCH then samples
  edges, KILL then samples a node.

Type head index layout:  0 = STOP, 1..B = batch_sizes[0..B-1], B+1 = KILL.

Cost: the N re-scorings (BATCH) are cheap policy forwards; the one expensive GPT-2
faith eval happens once per step in the env.
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
        self.n_sizes = len(self.batch_sizes)
        self.kill_idx = 1 + self.n_sizes              # last type index = KILL
        self.n_types = 2 + self.n_sizes               # STOP + sizes + KILL

        self.edge_embed = _mlp([edge_dim + 1, hidden, hidden])
        self.node_embed = _mlp([node_dim + 1, hidden, hidden])
        ctx_in = 2 * hidden + 2 * hidden + n_globals
        self.context = _mlp([ctx_in, hidden, hidden])
        self.edge_head = _mlp([hidden + hidden, hidden, 1])   # BATCH edge scores
        self.node_head = _mlp([hidden + hidden, hidden, 1])   # KILL node scores
        self.type_head = _mlp([hidden, hidden, self.n_types])
        self.value_head = _mlp([hidden, hidden, 1])

        # Init the type head to mimic the flat policy's healthy start: single-cut
        # (N=batch_sizes[0]) DOMINANT, STOP RARE; bigger batches and KILL explored
        # but not dominant (learned up over training, as KILL was in the flat run).
        with torch.no_grad():
            self.type_head[-1].bias.zero_()
            self.type_head[-1].bias[0] = -5.0         # STOP: ~never at init
            self.type_head[-1].bias[1] = 2.0          # smallest N (single cut): favored

    # ---- trunk + heads ----

    def _ctx_from(self, e, n, g):
        """Context from already-embedded edges/nodes + globals (the cheap part).

        Pulled out of _embed so the autoregressive batch loop can recompute the
        context after each cut WITHOUT re-running the edge/node MLPs (only the
        pooled reductions + a small MLP). Cutting an edge changes e.mean/e.amax
        and the alive-fraction global, so ctx must be recomputed -- but the
        per-edge/per-node embeddings are reused (see _embed_dead / act/evaluate).
        """
        pooled = torch.cat([e.mean(0), e.amax(0), n.mean(0), n.amax(0), g], dim=-1)
        return self.context(pooled)

    def _embed(self, obs):
        ef = obs["edge_features"]
        nf = obs["node_features"]
        ea = obs["edge_alive"].unsqueeze(-1)
        naf = obs["kill_alive_frac"].unsqueeze(-1)
        e = self.edge_embed(torch.cat([ef, ea], dim=-1))   # [K, H]
        n = self.node_embed(torch.cat([nf, naf], dim=-1))  # [M, H]
        ctx = self._ctx_from(e, n, obs["globals"])
        return ctx, e, n

    def _embed_dead(self, obs):
        """Edge embeddings with alive=0 for EVERY edge. Cutting edge i during the
        autoregressive loop just swaps row i of the live embeddings for this row i
        -- exactly what edge_embed([ef_i, 0]) would give -- so we never re-run the
        edge MLP per pick. Computed once per batch (lazily, only when N>1)."""
        ef = obs["edge_features"]
        zeros = torch.zeros_like(obs["edge_alive"]).unsqueeze(-1)
        return self.edge_embed(torch.cat([ef, zeros], dim=-1))   # [K, H]

    def _edge_logits(self, obs, ctx, e):
        ctx_e = ctx.unsqueeze(0).expand(e.size(0), -1)
        el = self.edge_head(torch.cat([e, ctx_e], dim=-1)).squeeze(-1)
        return el.masked_fill(~(obs["edge_alive"] > 0.5), MASK_VALUE)

    def _node_logits(self, obs, ctx, n):
        ctx_n = ctx.unsqueeze(0).expand(n.size(0), -1)
        nl = self.node_head(torch.cat([n, ctx_n], dim=-1)).squeeze(-1)
        return nl.masked_fill(~(obs["kill_alive_frac"] > 0.0), MASK_VALUE)

    def _type_logits(self, obs, ctx):
        tl = self.type_head(ctx)                            # [n_types]
        mask = torch.ones_like(tl, dtype=torch.bool)
        if not bool((obs["edge_alive"] > 0.5).any()):       # no alive edges -> no cut
            mask[1:1 + self.n_sizes] = False
        if not bool((obs["kill_alive_frac"] > 0.0).any()):  # no killable node
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
        """Sample (or argmax) a composite action. Returns (action, logp, entropy, value).

        action: {"type":"stop"} | {"type":"batch","size_idx",..,"edges":[..]}
                | {"type":"kill","node":idx}
        """
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

        # BATCH: autoregressively sample N edges, re-scoring between picks.
        # Edge/node embeddings are computed ONCE (e, n from _embed above); each cut
        # only swaps the cut edge's row to its alive=0 embedding (e_dead) and bumps
        # the alive-fraction global -- no per-pick edge/node MLP. Numerically
        # identical to re-embedding from scratch (verified by test_batch_policy).
        n_target = self.batch_sizes[t - 1]
        work = self._clone_dynamic(obs)
        ew = e
        e_dead = self._embed_dead(obs) if n_target > 1 else None
        edges: List[int] = []
        edge_ents: List[torch.Tensor] = []
        for j in range(n_target):
            cw = self._ctx_from(ew, n, work["globals"])
            el = self._edge_logits(work, cw, ew)
            if (el > MASK_VALUE / 2).sum() == 0:
                break
            d = Categorical(logits=el)
            pick = el.argmax() if greedy else d.sample()
            logp = logp + d.log_prob(pick)
            edge_ents.append(d.entropy())
            pi = int(pick.item())
            edges.append(pi)
            if e_dead is not None and j + 1 < n_target:
                ew = ew.index_copy(0, pick.view(1).long(), e_dead[pi:pi + 1])
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

        # BATCH -- teacher-forced replay of the stored edges. Same cached-embedding
        # trick as act() (cut row -> e_dead), so the per-pick logits (hence log_prob)
        # are identical to act's; the test asserts the exact match.
        sc = torch.as_tensor(action["size_idx"], dtype=torch.long, device=type_logits.device)
        logp = type_dist.log_prob(sc)
        ent_type = type_dist.entropy()
        work = self._clone_dynamic(obs)
        ew = e
        n_edges = len(action["edges"])
        e_dead = self._embed_dead(obs) if n_edges > 1 else None
        edge_ents: List[torch.Tensor] = []
        for j, pi in enumerate(action["edges"]):
            cw = self._ctx_from(ew, n, work["globals"])
            el = self._edge_logits(work, cw, ew)
            d = Categorical(logits=el)
            t = torch.as_tensor(pi, dtype=torch.long, device=el.device)
            logp = logp + d.log_prob(t)
            edge_ents.append(d.entropy())
            if e_dead is not None and j + 1 < n_edges:
                ew = ew.index_copy(0, t.view(1), e_dead[pi:pi + 1])
            self._mark_cut(work, pi)
        entropy = ent_type + (torch.stack(edge_ents).mean() if edge_ents else 0.0 * ent_type)
        return logp, entropy, value
