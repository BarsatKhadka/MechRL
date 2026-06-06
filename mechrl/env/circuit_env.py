"""CircuitEnv — Gym-style environment for the circuit-finding RL agent.

One episode = prune the top-K candidate edges of one task down to a minimal
faithful subgraph.

  reset()  -> pick a task, start with all K candidates alive, return observation
  step(a)  -> a in [0, K)   : CUT candidate edge a   (monotonic, never restored)
              a == K        : STOP, end episode, give terminal reward
              budget reached : auto-STOP

Faithfulness is computed by the AblationEngine (a forward pass through GPT-2
with the cut edges patched to corrupted activations). The CircuitReward turns
faithfulness changes into a scalar reward; the env reads reward.current_faith
for the observation so each step only triggers one forward pass.

The expensive per-task setup (load model, build graph, run EAP-IG prefilter,
build ablation engine) is done ONCE inside a TaskBundle. The env holds a list
of bundles and reset() just samples one and resets the cheap episode state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from mechrl.env.ablation import AblationEngine
from mechrl.env.graph import build_graph
from mechrl.env.prefilter import Prefilter
from mechrl.env.reward import CircuitReward
from mechrl.tasks.base import Task


# ---- node / edge feature encoding ----

def _node_type_onehot(name: str) -> Tuple[int, int, int, int]:
    """(is_attn, is_mlp, is_input, is_logits) for a node name."""
    if name.startswith("a"):
        return (1, 0, 0, 0)
    if name.startswith("m"):
        return (0, 1, 0, 0)
    if name == "input":
        return (0, 0, 1, 0)
    if name == "logits":
        return (0, 0, 0, 1)
    return (0, 0, 0, 0)


def _qkv_onehot(qkv: Optional[str]) -> Tuple[int, int, int]:
    """(is_q, is_k, is_v) for an edge's channel; (0,0,0) if not into attention."""
    return (int(qkv == "q"), int(qkv == "k"), int(qkv == "v"))


# Feature layouts (documented for the policy/encoder). All values are designed
# to be task-normalized so they mean the same thing on every task (key to
# transfer). EDGE features align with CUT actions (one row per candidate edge);
# NODE features align with KILL actions (one row per distinct parent node).
EDGE_FEATURE_NAMES = [
    "signed_norm_score",   # score / max|score| in this task         in [-1, 1]
    "rank_frac",           # 0 = most important edge, 1 = least       in [0, 1]
    "parent_layer_frac",   # parent.layer / 11                        in [0, 1]
    "child_layer_frac",    # child.layer  / 11                        in [0, 1]
    "layer_distance_frac", # (child.layer - parent.layer) / 11        in [-1, 1]
    "p_attn", "p_mlp", "p_input", "p_logits",   # parent type one-hot
    "c_attn", "c_mlp", "c_input", "c_logits",   # child  type one-hot
    "is_q", "is_k", "is_v",                     # channel one-hot
]
NODE_FEATURE_NAMES = [
    "node_layer_frac",        # node.layer / 11                       in [0, 1]
    "n_attn", "n_mlp", "n_input", "n_logits",   # node type one-hot
    "out_degree_frac",        # #candidate edges / max over nodes      in [0, 1]
    "agg_signed_norm_score",  # sum of its edge scores, normalized     in [-1, 1]
]
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)   # 16
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)   # 7
N_GLOBALS = 5                                # see CircuitEnv._observation


def build_features(engine, cand_edge_idx: torch.Tensor):
    """Build the static feature tensors + KILL grouping for one task.

    Returns
    -------
    edge_features : float[K, EDGE_FEATURE_DIM]   aligned with CUT actions 0..K-1
    node_features : float[M, NODE_FEATURE_DIM]   aligned with KILL actions 0..M-1
    parent_names  : list[str] length M           the KILL targets, sorted
    parent_groups : list[long]                   parent_groups[m] = candidate-local
                                                 indices of node m's edges

    All scores are normalized WITHIN this task (by max|score|), and rank is a
    fraction of K, so nothing carries the task's raw metric units. That is what
    lets a policy trained on one task act sensibly on another.
    """
    edge_list = engine.edge_list
    cand = cand_edge_idx.tolist()
    K = len(cand)

    # --- gather raw per-candidate info ---
    raw_scores, parents, children, qkvs = [], [], [], []
    for idx in cand:
        e = edge_list[idx]
        s = e.score
        raw_scores.append(s.item() if torch.is_tensor(s) else float(s))
        parents.append(e.parent)
        children.append(e.child)
        qkvs.append(e.qkv)
    raw = torch.tensor(raw_scores, dtype=torch.float32)

    # --- per-task normalization ---
    max_abs = raw.abs().max().clamp_min(1e-12)
    signed_norm = raw / max_abs                                  # in [-1, 1]
    # rank_frac: 0.0 for the largest |score|, 1.0 for the smallest
    order = torch.argsort(raw.abs(), descending=True)
    rank_frac = torch.empty(K, dtype=torch.float32)
    rank_frac[order] = torch.linspace(0.0, 1.0, K) if K > 1 else torch.zeros(1)

    # --- per-edge rows (aligned with CUT actions) ---
    edge_rows = []
    for i in range(K):
        p, c = parents[i], children[i]
        edge_rows.append(
            [
                float(signed_norm[i]),
                float(rank_frac[i]),
                p.layer / 11.0,
                c.layer / 11.0,
                (c.layer - p.layer) / 11.0,
                *_node_type_onehot(p.name),
                *_node_type_onehot(c.name),
                *_qkv_onehot(qkvs[i]),
            ]
        )
    edge_features = torch.tensor(edge_rows, dtype=torch.float32)

    # --- group candidate edges by PARENT node (the KILL targets) ---
    groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(K):
        groups[parents[i].name].append(i)
    parent_names = sorted(groups.keys())
    parent_groups = [torch.tensor(groups[n], dtype=torch.long) for n in parent_names]
    name_to_node = {parents[i].name: parents[i] for i in range(K)}

    # --- per-node rows (aligned with KILL actions) ---
    sizes = torch.tensor([g.numel() for g in parent_groups], dtype=torch.float32)
    max_size = sizes.max().clamp_min(1.0)
    agg = torch.tensor([float(raw[g].sum()) for g in parent_groups], dtype=torch.float32)
    max_abs_agg = agg.abs().max().clamp_min(1e-12)
    node_rows = []
    for m, name in enumerate(parent_names):
        node = name_to_node[name]
        node_rows.append(
            [
                node.layer / 11.0,
                *_node_type_onehot(name),
                float(sizes[m] / max_size),
                float(agg[m] / max_abs_agg),
            ]
        )
    node_features = torch.tensor(node_rows, dtype=torch.float32)

    return edge_features, node_features, parent_names, parent_groups


@dataclass
class TaskBundle:
    """Everything needed to run episodes on one task — built once, reused.

    Fields:
        task            : the Task instance (owns its GPT-2 model)
        engine          : AblationEngine for faithfulness queries
        candidate_mask  : bool[n_edges], True for the top-K EAP-IG candidates
        cand_edge_idx   : long[K], full-edge indices of the K candidates
        edge_features   : float[K, EDGE_FEATURE_DIM], static per-edge features
        node_features   : float[M, NODE_FEATURE_DIM], static per-KILL-node features
        n_candidates    : K
        parent_names    : list[str] length M, distinct parent nodes (KILL targets)
        parent_groups   : list[long], parent_groups[m] = candidate-local indices
                          (in [0,K)) of edges whose parent is parent_names[m]
    """
    task: Task
    engine: AblationEngine
    candidate_mask: torch.Tensor
    cand_edge_idx: torch.Tensor
    edge_features: torch.Tensor
    node_features: torch.Tensor
    n_candidates: int
    parent_names: List[str]
    parent_groups: List[torch.Tensor]

    @classmethod
    def build(
        cls,
        task: Task,
        k: int = 3000,
        ig_steps: int = 5,
        prefilter_batch_size: int = 10,
    ) -> "TaskBundle":
        """Run the one-time expensive setup for a task."""
        graph = build_graph(task.model)
        # KL faithfulness: reproduce the full model's output distribution (caps at
        # 1.0, keeps suppressors — no faith>1.0 overshoot). See reward-loop.md.
        engine = AblationEngine(task, graph, metric_type="kl")

        pref = Prefilter(task, graph, ig_steps=ig_steps)
        pref.compute(batch_size=prefilter_batch_size)
        candidate_mask = pref.candidate_mask(k)
        cand_edge_idx = candidate_mask.nonzero(as_tuple=True)[0]
        n_candidates = int(cand_edge_idx.numel())

        # Build static edge/node features + KILL grouping (all task-normalized).
        edge_features, node_features, parent_names, parent_groups = build_features(
            engine, cand_edge_idx
        )

        return cls(
            task=task,
            engine=engine,
            candidate_mask=candidate_mask,
            cand_edge_idx=cand_edge_idx,
            edge_features=edge_features,
            node_features=node_features,
            n_candidates=n_candidates,
            parent_names=parent_names,
            parent_groups=parent_groups,
        )


class CircuitEnv:
    """Gym-style env over a list of TaskBundles.

    Action space is flat-discrete of size (K + M + 1):
        0 .. K-1         -> CUT that candidate edge
        K .. K+M-1       -> KILL parent node (cut all its alive outgoing candidates)
        K+M              -> STOP
    where K = n_candidates and M = number of distinct parent nodes.

    KILL is outgoing-only: it cuts every still-alive candidate edge whose PARENT
    is that node, silencing the node's output in one action. Sparsity reward is
    flat (the multi-edge cut gets one +sparsity_weight, same as a single CUT);
    the removed edges are rewarded through terminal minimality.

    Note: when bundles differ in K or M, the action dimension varies per episode.
    Read `env.action_dim` after reset() to size the policy head, or use bundles
    with shared K (the normal case: all K=3000).
    """

    def __init__(
        self,
        bundles: List[TaskBundle],
        step_budget: int = 500,
        faith_threshold: float = 0.8,
        threshold_penalty: float = 3.0,
        invalid_penalty: float = -0.01,
        seed: int = 0,
        minimality_weight: float = 1.0,
    ):
        if not bundles:
            raise ValueError("CircuitEnv needs at least one TaskBundle.")
        self.bundles = bundles
        self.step_budget = step_budget
        self.faith_threshold = faith_threshold
        self.threshold_penalty = threshold_penalty
        self.invalid_penalty = invalid_penalty
        self.minimality_weight = minimality_weight
        self._rng = torch.Generator().manual_seed(seed)

        # Episode state (set by reset)
        self.bundle: Optional[TaskBundle] = None
        self.reward: Optional[CircuitReward] = None
        self.mask: Optional[torch.Tensor] = None          # bool[n_edges] working mask
        self.alive: Optional[torch.Tensor] = None         # bool[K] candidate alive flags
        self.n_candidates: int = 0
        self.steps: int = 0
        self.done: bool = True
        self.faith_start: float = 0.0
        self._last_faith_delta: float = 0.0

    # ---- lifecycle ----

    def reset(self, bundle_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Start a new episode. Samples a bundle (random if not specified)."""
        if bundle_idx is None:
            bundle_idx = int(torch.randint(len(self.bundles), (1,), generator=self._rng).item())
        self.bundle = self.bundles[bundle_idx]
        self.bundle_idx = bundle_idx

        self.n_candidates = self.bundle.n_candidates
        self.n_kill = len(self.bundle.parent_names)
        # Start with all candidates alive (non-candidate edges stay cut).
        self.mask = self.bundle.candidate_mask.clone()
        self.alive = torch.ones(self.n_candidates, dtype=torch.bool)
        self.steps = 0
        self.done = False

        self.reward = CircuitReward(
            self.bundle.engine,
            faith_threshold=self.faith_threshold,
            threshold_penalty=self.threshold_penalty,
            invalid_penalty=self.invalid_penalty,
            step_budget=self.step_budget,
            minimality_weight=self.minimality_weight,
        )
        self.reward.begin_episode(self.bundle.candidate_mask)
        self.faith_start = float(self.reward.current_faith)
        self._last_faith_delta = 0.0

        return self._observation()

    def step(self, action) -> Tuple[Dict[str, torch.Tensor], float, bool, dict]:
        """Apply one action. Returns (obs, reward, done, info).

        Two action formats are accepted:
          - int (legacy flat):  0..K-1 = CUT edge, K..K+M-1 = KILL parent, K+M = STOP
          - dict (batch-cut):   {"type":"stop"} or
                                {"type":"batch","size_idx":int,"edges":[cand-local idxs]}
        """
        if isinstance(action, dict):
            return self._step_batch(action)
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        self.steps += 1
        K, M = self.n_candidates, self.n_kill
        faith_pre = float(self.reward.current_faith)

        # STOP
        if action == K + M:
            reward = self.reward.terminal(self.mask)
            self.done = True
            info = {"reason": "stop", "faith": self.reward.current_faith,
                    "kept": int(self.alive.sum().item()), "steps": self.steps,
                    "n_cut_this_step": 0}
            return self._observation(), reward, True, info

        # Decode CUT vs KILL and apply.
        if 0 <= action < K:
            reason = "cut"
            valid = bool(self.alive[action])
            n_cut = 0
            if valid:
                self.alive[action] = False
                self.mask[self.bundle.cand_edge_idx[action]] = False
                n_cut = 1
        elif K <= action < K + M:
            reason = "kill"
            group = self.bundle.parent_groups[action - K]      # candidate-local indices
            alive_in_group = group[self.alive[group]]          # only those still alive
            n_cut = int(alive_in_group.numel())
            valid = n_cut > 0
            if valid:
                self.alive[alive_in_group] = False
                self.mask[self.bundle.cand_edge_idx[alive_in_group]] = False
        else:
            raise IndexError(f"action {action} out of range [0, {K + M}]")

        # Flat sparsity: one +sparsity_weight regardless of how many edges were cut.
        reward = self.reward.step(self.mask, valid_action=valid)
        if not valid:
            reason = "invalid"
        self._last_faith_delta = float(self.reward.current_faith) - faith_pre

        # Auto-STOP on budget exhaust: fold in the terminal reward.
        if self.steps >= self.step_budget:
            reward = reward + self.reward.terminal(self.mask)
            self.done = True
            reason = "budget"

        info = {"reason": reason, "faith": self.reward.current_faith,
                "kept": int(self.alive.sum().item()), "steps": self.steps,
                "valid": valid, "n_cut_this_step": n_cut}
        return self._observation(), reward, self.done, info

    def _step_batch(self, action: dict) -> Tuple[Dict[str, torch.Tensor], float, bool, dict]:
        """Apply a batch-cut / stop action (see BatchCutPolicy).

        A batch cuts several candidate edges, then scores faith ONCE — so the
        whole batch costs a single GPT-2 forward (the reward's faith eval).
        """
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        self.steps += 1

        if action["type"] == "stop":
            reward = self.reward.terminal(self.mask)
            self.done = True
            info = {"reason": "stop", "faith": self.reward.current_faith,
                    "kept": int(self.alive.sum().item()), "steps": self.steps,
                    "valid": True, "n_cut_this_step": 0}
            return self._observation(), reward, True, info

        # Apply cuts, then ONE faith eval. BATCH = the chosen edges; KILL = all of
        # one parent node's still-alive candidate edges.
        faith_pre = float(self.reward.current_faith)
        n_cut = 0
        if action["type"] == "batch":
            for idx in action["edges"]:
                if self.alive[idx]:
                    self.alive[idx] = False
                    self.mask[self.bundle.cand_edge_idx[idx]] = False
                    n_cut += 1
            reason = "batch"
        elif action["type"] == "kill":
            group = self.bundle.parent_groups[action["node"]]   # candidate-local indices
            alive_in_group = group[self.alive[group]]
            n_cut = int(alive_in_group.numel())
            if n_cut > 0:
                self.alive[alive_in_group] = False
                self.mask[self.bundle.cand_edge_idx[alive_in_group]] = False
            reason = "kill"
        else:
            raise ValueError(f"unknown batch action type: {action['type']!r}")

        valid = n_cut > 0
        reward = self.reward.step(self.mask, valid_action=valid)
        self._last_faith_delta = float(self.reward.current_faith) - faith_pre

        if not valid:
            reason = "invalid"
        if self.steps >= self.step_budget:
            reward = reward + self.reward.terminal(self.mask)
            self.done = True
            reason = "budget"
        info = {"reason": reason, "faith": self.reward.current_faith,
                "kept": int(self.alive.sum().item()), "steps": self.steps,
                "valid": valid, "n_cut_this_step": n_cut}
        return self._observation(), reward, self.done, info

    # ---- observation ----

    def _observation(self) -> Dict[str, torch.Tensor]:
        """Current state the policy sees.

        Static (same every step, from the bundle):
            edge_features : float[K, EDGE_FEATURE_DIM]  per-candidate features
            node_features : float[M, NODE_FEATURE_DIM]  per-KILL-node features

        Dynamic (change as the agent acts):
            edge_alive      : float[K]  1.0 if candidate still in graph else 0.0
                              (also the validity mask for CUT actions)
            kill_alive_frac : float[M]  fraction of each node's edges still alive
                              (>0 means the KILL action is still valid)
            globals         : float[5]  [step_fraction, current_faith,
                              alive_fraction, faith_delta_last_step, faith_start]
        """
        alive_f = self.alive.float()
        # Fraction of each parent group's edges still alive (0.0 => KILL invalid).
        kill_alive_frac = torch.tensor(
            [float(self.alive[g].float().mean()) for g in self.bundle.parent_groups],
            dtype=torch.float32,
        )
        globals_ = torch.tensor(
            [
                self.steps / max(1, self.step_budget),
                float(self.reward.current_faith),
                alive_f.mean().item(),
                self._last_faith_delta,
                self.faith_start,
            ],
            dtype=torch.float32,
        )
        return {
            "edge_features": self.bundle.edge_features,
            "node_features": self.bundle.node_features,
            "edge_alive": alive_f,
            "kill_alive_frac": kill_alive_frac,
            "globals": globals_,
        }

    @property
    def action_dim(self) -> int:
        """Number of actions this episode: K candidates + M kill-nodes + 1 STOP."""
        return self.n_candidates + self.n_kill + 1
