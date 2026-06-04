# Reward Loop — design log for the circuit-finding RL agent

A running record of how we designed the reward, every failure mode we hit, what
each one taught us, and the open problems + levers. This is the "research grind"
log — most of it won't make the paper, but it's the actual work and it keeps us
from re-walking dead ends.

---

## The goal

Train an RL agent that prunes GPT-2's computational graph (top-K=3000 candidate
edges per task) down to the **smallest subgraph that still does the task** — i.e.
the **minimal faithful circuit** — and does it **with few interventions** (the
transfer story). On a single task (IOI) we just need: *does it learn to prune to
a small circuit while staying faithful?*

---

## The substrate: faithfulness

```
faith(mask) = (score(mask) - corrupted_baseline) / (full_baseline - corrupted_baseline)
```
- `faith = 1.0` → subgraph matches the full model.
- `faith = 0.0` → no better than cutting everything (corrupted floor).
- `faith > 1.0` → subgraph BEATS the full model (happens when harmful edges, e.g.
  the negative name-mover `a10.h7→logits`, are cut). This is real headroom EAP-
  threshold can't reach.
- Top-3000 IOI starts at **faith ≈ 0.72**.

Each step the agent CUTs one edge or KILLs a whole node (cuts all its outgoing
candidate edges) or STOPs. Cutting = patch that edge with the corrupted-prompt
activation. Faithfulness is measured by a real GPT-2 forward pass per step
(corrupted pass cached per task → ~1.7× speedup; bit-identical).

---

## Reward v1 — dense Δfaith + sparsity + faith×minimality

```
per-step (valid)   = Δfaith + sparsity_weight(0.001)
per-step (invalid) = invalid_penalty(-0.01)
terminal (STOP)    = max(0, faith) × (1 - kept/n_candidates)
```

### Run A — budget=400
- **Result:** CONVERGED. faith stable **1.3–1.5** (up to 1.51 — beats full GPT-2
  by 51%, by cutting harmful negative-mover edges). Validates the thesis headroom.
- **BUT:** `kept` pinned at **exactly 2600 = 3000 − budget(400)**.
- **Diagnosis 1 (the faith-park):** the per-step `Δfaith` terms **sum over the
  episode to `faith_final − faith_start`**. So the return secretly contains a big
  bonus for *maximizing final faith* — which rewards "cut the harmful edges, keep
  everything else." The terminal `faith×minimality` (~0.18 at kept 2600) was far
  too weak to pull it toward a small circuit. **The reward was fighting itself.**
- **Diagnosis 2 (budget wall):** with single cuts, `kept` can't go below
  `3000 − budget`. Budget = an external size dial.

### Run B — budget=1500 (overnight)
- **Result:** TOO SLOW (~10 min/iter; 62 iters in 10 h). `kept` reverted to
  **1500 = 3000 − budget** (budget wall again, agent back to single-cutting).
  Learning *slower* (faith only ~0.5 by iter 62 vs 1.4 by iter 140 at budget 400).
- **Diagnosis 3 (speed):** per-step cost is one GPT-2 forward (~0.4 s). Episode
  length = budget, so big budget = many forwards/episode = slow. **Every step
  pays the forward.**
- **Diagnosis 4 (budget = size dial — user's catch):** `kept ≈ 3000 − budget`
  means *we* are setting the circuit size, not the agent. The per-step sparsity
  bonus makes "cut one more edge" always worth +0.001, so the agent cuts until
  the budget runs out instead of STOPping at the natural minimal circuit. **The
  budget should be a generous safety cap; the agent should choose the size.**

### What v1 taught us
1. The agent CAN identify and exploit harmful edges (faith > 1.0). ✓ thesis.
2. Per-step Δfaith secretly maximizes faith → don't reward faith above the bar.
3. Budget must not be the size dial → make it a non-binding cap + a learned STOP.
4. Per-step sparsity drives "cut-to-budget" → drop it.
5. Speed is bounded by the per-step forward → short episodes (KILL + STOP) are
   the only way to make it fast AND let the agent self-size.

---

## Reward v2 — threshold potential (current)

Encode the *literature's* definition (smallest circuit with faith ≥ τ) as a
potential and reward its per-step change:

```
Φ(faith, kept) = minimality − λ · max(0, τ − faith)
                 minimality = 1 − kept / n_candidates

per-step (valid)   = Φ(after) − Φ(before)    # dense, telescopes to the objective
per-step (invalid) = invalid_penalty(-0.01)
terminal (STOP)    = 0                         # Φ already accumulated
```
Defaults: **τ = 0.8** (retain 80% of behaviour), **λ = 3**.

### Why this fixes v1's problems
- **No faith-park:** above τ, faith is irrelevant (0.9 and 1.4 score the same) →
  reward is pure minimality → no incentive to park at 1.4.
- **Self-sizing + learned STOP:** continuing past the optimum gives negative ΔΦ
  (cutting below τ is penalised), so the agent learns to STOP at the smallest
  circuit with faith ≈ τ. Budget becomes a non-binding cap.
- **KILL is rewarded proportionally:** a safe KILL removing `k` edges gives
  `ΔΦ ≈ k/n_candidates` (~40× a single cut), while a bad KILL (faith < τ) is
  sharply penalised. So the reward already *favours* good KILLs.
- **Curriculum for free:** start (faith 0.72 < τ) is below the bar, so the agent
  is first rewarded for cutting harmful edges to clear τ, *then* for minimality.

### Validation (IOI, offline, bit-checked)
- start faith 0.72 < τ → Φ_start = −0.24 ✓
- cut helper `a10.h7→logits`: faith 0.72→0.96, r = **+0.24** ✓
- cut 200 most-important: faith → −0.02, r = **−2.14** ✓
- cut 200 lowest-ranked (safe): faith holds, r = **+0.09** ✓
- objective(top-1k) > objective(top-3k) ✓

### Run C — budget=400 (Aquaman, 2× RTX 3070)
- **Result:** STUCK. `ent` plateaued/oscillated (8.05 → 7.4 → back to 7.8),
  `kept` not dropping (~2700), `return` noisy around 0. Briefly flirted with a
  **"do-nothing / STOP-early" collapse** (episodes spiked to 25–37, kept ~2980,
  return ~0 — i.e. barely cut, STOP, dodge the penalty).
- **Diagnosis 5 (risk aversion):** the λ=3 penalty makes cutting scary → the
  "safe" move is to barely cut and STOP (return ≈ 0 beats −2 from a bad cut).
- **Diagnosis 6 (noisy gradient):** each update used only 1–2 episodes, and
  returns swing wildly (−2.0 to +0.7). Noisy reward × few samples = noisy gradient
  that points different ways each iter → **entropy oscillates, no progress.**
  (This is a signal-to-noise problem, NOT a learning-rate problem — bumping LR
  would amplify the noise. Reflex-rejected.)
- **Diagnosis 7 (exploration — the deep one):** `kept` *never* drops below ~2700,
  so the agent **never visits a small-circuit state** → never experiences that
  small faithful circuits give big reward → no gradient toward the goal. It can't
  learn a policy it never samples.

### Run D — budget=150, num_steps=512 (current, in progress)
- **Hypothesis:** shorter episodes pack **more episodes per update** (~3–5 vs 1–2)
  → cleaner gradient — at the **same speed** (speed = num_steps = forwards/iter,
  *not* episode count). Also: a 150-step budget can't single-cut deep (floor
  ~2850), so it **pressures the agent toward KILL.**
- Early (iter 1–4): episodes 3–5/iter ✓, `kept` already dipping to 2657 (< 2850 →
  KILL being sampled) ✓. `ent ≈ 8.0` (too early).
- **Watching:** `ent` falling *steadily* (gradient fixed?) + `kept` dropping
  below ~2850 (KILL used on purpose?).

---

## Key relationships to remember

```
speed per iter     = num_steps                 (each step = 1 GPT-2 forward)
episodes per iter  = num_steps / episode_len   (episode_len ≈ step_budget)
=> more episodes at SAME speed  ==>  shorter episodes (lower budget), NOT bigger num_steps
kept_min via single-cuts        =  3000 − budget   (only KILL beats this)
return                          =  Φ_final − Φ_start   (telescoped ΔΦ)
```

---

## Reward v3 turning point — KL faithfulness (the metric was the problem)

The whole faith>1.0 / suppressor mess came from the **metric**, not the reward.
We used each task's natural metric (logit-diff) — a SINGLE scalar, which a circuit
can "beat" (faith → 1.4) by cutting the negative heads. Switching faithfulness to
**KL divergence** — `KL(full-model distribution ‖ circuit distribution)` at the
prediction position — dissolves it:
- KL ≥ 0, KL=0 = circuit reproduces the model. So **faith caps at 1.0** (no overshoot).
- Cutting a suppressor makes the circuit DIVERGE from the model → KL up → faith DOWN
  → **suppressors are kept automatically**, no hand-coded rule.
- It's the metric ACDC uses → clean baseline comparison.
- `faith = 1 − KL(circuit)/KL(all-cut)` (same normalization formula; full=KL(full‖full)=0).

**Measured on IOI (test_kl_faith.py):**
- KL full baseline 0.0; KL all-cut baseline 4.65.
- **KL-faith top-3k = 0.95** (vs logit-diff 0.72). HIGHER, because the 0.72 was
  artificially depressed by the suppressor — the full model *also* suppresses, so
  the top-3k matches the model 95% in KL terms. **K=3000 is plenty; no bigger K needed.**
- **The flip:** cut a10.h7 → KL-faith 0.95→0.92 (DROPS, kept) vs logit-diff 0.72→0.96
  (rises, cut). The negative-head problem is gone structurally.

**Why this is the right objective:** "smallest subset whose OUTPUT DISTRIBUTION
matches the model (KL-faith → ~0.95), keeping movers AND suppressors, cutting only
redundancy" = the *model's* circuit (what published-circuit validation wants),
not the best *task* subnetwork. Implemented in `ablation.py` (metric_type="kl",
caches the full-model logits once per task) + `circuit_env.py` (TaskBundle uses it).

**Implication for the reward:** the threshold reward (v2) now works correctly with
KL — no overshoot to handle. Set τ ≈ 0.9 (just under the 0.95 reference): prune while
KL-faith stays ≥ τ. No band-cap, no suppressor rule needed.

---

## Open problem

The agent isn't learning to prune deep. Two intertwined causes:
1. **Noisy gradient** (few episodes/update) — being tested via budget=150.
2. **Exploration** — it avoids KILL (~5% of actions, random kills hit important
   nodes → learns to fear them), so it never reaches small-circuit states.

---

## Levers forward (in order we'd try them)

1. **Cleaner gradient** — shorter episodes / more episodes per update. *(testing
   now: budget=150.)*
2. **Incentivize KILL = fix EXPLORATION, not reward** (reward already favours good
   KILLs). Ranked:
   - **(a) Bias KILL up at init** — give the policy's node-head a positive bias so
     KILL starts at ~25–30% action mass (vs ~5%). It tries kills, learns which are
     safe from the node features (`agg_signed_score`). Cleanest; reversible.
   - **(b) Safe-kill curriculum** — early on only allow killing low-`agg_score`
     (useless) nodes, so it learns "kills are good" before facing risky ones.
   - **(c) Lower λ** — make the occasional bad kill less catastrophic → less risk-
     averse about *trying* kills.
   - **DON'T:** add a flat KILL reward bonus → causes reckless killing.
3. **Warm-start / behavior cloning (the escape hatch).** If cold-start exploration
   stays the wall: behavior-clone the policy from **ACDC / EAP-threshold
   trajectories** (which DO find good circuits), then let PPO refine. Sidesteps
   "never visits the reward" entirely, and arguably strengthens the paper's
   "RL refines/transfers a known skill" framing. This was always the planned
   insurance for exactly this moment.

---

## Config knobs (sbatch / train_agent.py)

| knob | meaning | default |
|---|---|---|
| `--step-budget` / BUDGET | max actions/episode (safety cap) | 400 |
| `--num-steps` / STEPS | rollout length (= speed); defaults to BUDGET | 512 |
| `--faith-threshold` / TAU | τ — faithfulness bar | 0.8 |
| `--threshold-penalty` / PENALTY | λ — threshold hardness | 3.0 |
| `--lr` / LR | learning rate | 2.5e-4 |
| `--total-iterations` / ITERS | PPO iterations | 500 |

Run direct (no SLURM, e.g. Aquaman):
```bash
CUDA_VISIBLE_DEVICES=0 python -m mechrl.train.train_agent \
    --tasks IOITask --device cuda --step-budget 150 --num-steps 512 \
    --total-iterations 600 --out runs
```

---

## How to read the logs

| column | meaning | healthy direction |
|---|---|---|
| `return` | real total reward this episode (= Φ_final − Φ_start) | up, positive |
| `faith` | faithfulness at episode end | rises to ~τ and holds |
| `kept` | edges remaining (3000 = none cut) | **drops** |
| `ent` | policy entropy (ln 3151 ≈ 8.05 = uniform) | **falls steadily** |
| `episodes` | episodes completed this iter | higher = more samples (good) |
| `kl` | policy change per update | small, stable |
| `expl_var` | how well critic predicts returns (0=useless,1=perfect) | rises |

**Collapse signal:** `kept` stuck ~3000 + `return` pinned ~0 = "do nothing."
**Win signal:** `kept` drops AND `faith` holds ≈ τ AND `return` climbs.
