"""CircuitReward — reward function for the circuit-finding RL agent.

Per-step (dense, potential-based):
    r_t = Δfaithfulness + sparsity_bonus + invalid_penalty

Terminal (sparse, on STOP or budget exhaust):
    r_T = max(0, faith_final) × (1 - kept_edges / n_candidates)

Faith is normalized to [0,1] via AblationEngine.faithfulness(), so reward
scale is consistent across all tasks and their different natural metrics.
"""

from __future__ import annotations

import torch

from mechrl.env.ablation import AblationEngine


class CircuitReward:
    """Compute per-step and terminal rewards for one episode.

    Call `begin_episode` at the start of each episode to set n_candidates
    and cache the pre-episode faithfulness. Then call `step` after each
    action and `terminal` when the episode ends.

    Parameters
    ----------
    engine : AblationEngine
        Shared ablation engine (already has task + graph wired up).
    sparsity_weight : float
        Bonus per edge cut. Kept small (default 0.001) so it nudges
        toward smaller circuits without dominating the faith signal.
    invalid_penalty : float
        Penalty when action targets an edge outside the candidate set
        or already cut. Negative, default -0.01.
    step_budget : int
        Max steps before the env auto-terminates. Stored here for reference;
        the env loop enforces it.
    """

    def __init__(
        self,
        engine: AblationEngine,
        sparsity_weight: float = 0.001,
        invalid_penalty: float = -0.01,
        step_budget: int = 500,
    ):
        self.engine = engine
        self.sparsity_weight = sparsity_weight
        self.invalid_penalty = invalid_penalty
        self.step_budget = step_budget

        self.n_candidates: int = 0
        self._faith_before: float = 0.0

    @property
    def current_faith(self) -> float:
        """Latest faithfulness value tracked by the reward (set by begin_episode
        and updated by each valid step). The env reads this for the observation
        so it doesn't trigger a second forward pass per step."""
        return self._faith_before

    # ---- Episode lifecycle ----

    def begin_episode(self, candidate_mask: torch.Tensor) -> None:
        """Call once at episode start with the prefilter candidate mask.

        Caches n_candidates and computes the initial faithfulness so the
        first step's Δfaith is relative to the starting state.
        """
        self.n_candidates = int(candidate_mask.sum().item())
        self._faith_before = float(self.engine.faithfulness(candidate_mask))

    # ---- Per-step reward ----

    def step(
        self,
        mask_after: torch.Tensor,
        valid_action: bool,
    ) -> float:
        """Reward for a single CUT/KILL action.

        Parameters
        ----------
        mask_after : torch.Tensor
            Boolean edge mask AFTER the action has been applied.
        valid_action : bool
            False if the agent tried to cut an already-cut or out-of-candidate
            edge. Gives `invalid_penalty` and does not update faith tracking.

        Returns
        -------
        float
            Scalar reward for this step.
        """
        if not valid_action:
            return self.invalid_penalty

        faith_after = float(self.engine.faithfulness(mask_after))
        delta_faith = faith_after - self._faith_before
        self._faith_before = faith_after

        return delta_faith + self.sparsity_weight

    # ---- Terminal reward ----

    def terminal(self, final_mask: torch.Tensor) -> float:
        """Terminal reward on STOP or budget exhaust.

        r_T = max(0, faith_final) × (1 - kept_edges / n_candidates)

        Both faith and minimality must be nonzero to score: cutting
        everything → faith=0; keeping everything → minimality=0.

        Parameters
        ----------
        final_mask : torch.Tensor
            Boolean edge mask at episode end.

        Returns
        -------
        float
            Terminal scalar reward.
        """
        faith = max(0.0, float(self.engine.faithfulness(final_mask)))
        kept = int(final_mask.sum().item())
        if self.n_candidates == 0:
            minimality = 0.0
        else:
            # Clip to [0,1]: if final_mask has more True entries than n_candidates
            # (e.g. includes non-candidate edges), don't penalise negatively.
            minimality = max(0.0, 1.0 - kept / self.n_candidates)
        return faith * minimality
