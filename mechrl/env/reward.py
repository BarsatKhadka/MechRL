"""CircuitReward — threshold-based reward for the circuit-finding RL agent.

We want the SMALLEST circuit whose faithfulness stays above a threshold τ — the
same thing ACDC/EAP report, so the comparison is apples-to-apples. We encode that
as a potential Φ and reward its per-step change (dense, telescopes to the objective):

    Φ(faith, kept) = minimality − λ · max(0, τ − faith)
                     minimality = 1 − kept / n_candidates

    per-step (valid)   = Φ(after) − Φ(before)
    per-step (invalid) = invalid_penalty
    terminal (STOP)    = 0   (Φ is already accumulated through the per-step terms)

Behaviour:
  - below τ: cutting harmful edges raises faith → large +ΔΦ (clear the bar first)
  - above τ: faith is irrelevant → reward is pure minimality (cut anything safe)
  - cutting below τ: sharp penalty (don't)
  - KILL: ΔΦ ≈ (edges removed)/n_candidates, so killing a USELESS node (faith stays
    ≥ τ) is rewarded ~proportionally to its size, while killing a load-bearing node
    is penalised — directly incentivising intelligent bulk pruning.
  - STOP: continuing past the optimum gives negative ΔΦ, so the agent learns to stop
    at the smallest circuit with faith ≈ τ. Budget becomes a non-binding safety cap.

Faith is normalized to [0,1]-ish via AblationEngine.faithfulness() (can exceed 1
when harmful edges are cut), so τ means the same thing on every task.
"""

from __future__ import annotations

import torch

from mechrl.env.ablation import AblationEngine


class CircuitReward:
    """Threshold-potential reward for one episode.

    Parameters
    ----------
    engine : AblationEngine
        Shared ablation engine (task + graph wired up).
    faith_threshold : float
        τ — the faithfulness bar the circuit must clear (default 0.8 = retain 80%
        of full-model behaviour).
    threshold_penalty : float
        λ — how hard the threshold is. Larger → faith below τ is punished more,
        making τ behave more like a hard constraint (default 3.0).
    invalid_penalty : float
        Penalty for an action that cuts an already-cut / out-of-candidate edge.
    step_budget : int
        Max steps before the env auto-terminates (a safety cap; the agent should
        STOP well before this once it reaches the optimum). Stored for reference.
    """

    def __init__(
        self,
        engine: AblationEngine,
        faith_threshold: float = 0.8,
        threshold_penalty: float = 3.0,
        invalid_penalty: float = -0.01,
        step_budget: int = 500,
        minimality_weight: float = 1.0,
    ):
        self.engine = engine
        self.tau = faith_threshold
        self.lam = threshold_penalty
        self.invalid_penalty = invalid_penalty
        self.step_budget = step_budget
        # w: how strongly minimality (cutting edges) is rewarded relative to the
        # faith penalty. Default 1.0 = original behavior. Raise it if the agent is
        # too risk-averse and plateaus with too many edges kept.
        self.w = minimality_weight

        self.n_candidates: int = 0
        self._faith_before: float = 0.0
        self._phi_before: float = 0.0

    @property
    def current_faith(self) -> float:
        """Latest faithfulness value (set by begin_episode, updated each valid step).
        The env reads this for the observation so it doesn't trigger a 2nd forward."""
        return self._faith_before

    # ---- potential ----

    def _minimality(self, kept: int) -> float:
        if self.n_candidates == 0:
            return 0.0
        return max(0.0, 1.0 - kept / self.n_candidates)

    def _potential(self, faith: float, kept: int) -> float:
        return self.w * self._minimality(kept) - self.lam * max(0.0, self.tau - faith)

    # ---- episode lifecycle ----

    def begin_episode(self, candidate_mask: torch.Tensor) -> None:
        """Call once at episode start with the prefilter candidate mask."""
        kept = int(candidate_mask.sum().item())
        self.n_candidates = kept
        self._faith_before = float(self.engine.faithfulness(candidate_mask))
        self._phi_before = self._potential(self._faith_before, kept)

    # ---- per-step reward ----

    def step(self, mask_after: torch.Tensor, valid_action: bool) -> float:
        """Reward for one CUT/KILL action = ΔΦ (or invalid_penalty)."""
        if not valid_action:
            return self.invalid_penalty

        faith_after = float(self.engine.faithfulness(mask_after))
        kept = int(mask_after.sum().item())
        phi_after = self._potential(faith_after, kept)

        reward = phi_after - self._phi_before
        self._faith_before = faith_after
        self._phi_before = phi_after
        return reward

    # ---- terminal ----

    def terminal(self, final_mask: torch.Tensor) -> float:
        """STOP / budget-exhaust. The objective Φ is already accumulated through the
        per-step ΔΦ terms, so STOP itself adds nothing."""
        return 0.0

    # ---- introspection (for logging) ----

    def objective(self, final_mask: torch.Tensor) -> float:
        """The current objective value Φ — useful for logging 'how good is this
        circuit' independent of the per-step shaping."""
        faith = float(self.engine.faithfulness(final_mask))
        kept = int(final_mask.sum().item())
        return self._potential(faith, kept)
