# Working Environment Design

The RL environment for circuit discovery in GPT-2 small. This doc explains what the environment is, how it works step by step, and why the design choices were made. Read this before touching the code.

---

## What this environment is

A Gymnasium environment that exposes GPT-2 small's internal computation graph as an RL problem. The agent's job: starting from the full graph, prune edges until only a small faithful subgraph remains — the "circuit" that explains the model's behavior on a given task.

The environment wraps ACDC's existing graph representation (`TLACDCCorrespondence` from the Conmy et al. codebase) so we inherit a battle-tested DAG and ablation machinery rather than rebuild it.

**Project context**: this environment is the foundation for an RL agent that learns transferable circuit-finding skill. Trained on four templated tasks, tested on a held-out fifth, and later extended to OpenWebText. See `whole-plan.md` for the full project plan.

---

## Mental model: GPT-2 as a DAG

GPT-2 small has 12 layers, 12 heads per layer, plus MLPs. ACDC represents it as a graph where:

- **Nodes** are `(hook_point, slice_index)` pairs at TransformerLens hook points. Per layer this gives:
  - 12 heads × 7 nodes each (`hook_result`, `hook_q/k/v`, `hook_q/k/v_input`)
  - 2 MLP nodes (`hook_mlp_out`, `hook_mlp_in`)
- Plus the embedding node and final `resid_post`.
- Total: ~1,030 nodes, ~32,000 edges.

**Edges have three types** (this matters for ablation semantics):

- `ADDITION` — residual-stream writes. Cutting one means "remove this writer's contribution from this reader's residual sum."
- `DIRECT_COMPUTATION` — a node is a deterministic function of one parent (e.g. `hook_q` from `hook_q_input`). Cutting means replacing the parent's value with a corrupted activation.
- `PLACEHOLDER` — bookkeeping edges that keep the graph connected but aren't independently ablatable. Always present.

The agent only ever acts on `ADDITION` and `DIRECT_COMPUTATION` edges. Placeholders are invisible to it.

A **circuit** is a subset of these edges that's enough to reproduce the full model's output on a task. Finding the minimum faithful subset is the agent's job.

---

## How the environment works, step by step

### Episode start (env.reset)

1. **Pick a task** at random from the configured pool (IOI, greater-than, induction, docstring, copy-suppression). Each task knows how to generate its own prompts.
2. **Generate a batch of prompts** — a clean batch (real prompts) and a corrupted batch (same structure, scrambled content). Typically 32 prompts.
3. **Pre-filter the edges**. Run attribution patching: one clean forward pass, one corrupted forward pass, one backward pass. From the three, compute for every edge:

   ```
   score[e] = | gradient[e] · (corrupted_activation[e] − clean_activation[e]) |
   ```

   Sort and keep the top ~3,000 edges as **candidates**. The other ~29,000 are frozen as "kept" for the whole episode — the agent never decides about them. This collapses the action space from intractable to tractable in three forward/backward passes.

4. **Record the full model's logits** on the clean batch. This is the gold standard. Every later subgraph is compared against it for faithfulness.
5. **Initialize state**: edge mask (all candidates present), active-node list, step counter.

Return the initial observation.

### Each step (env.step)

1. **Receive the agent's action**. Action is factored into three parts:
   - `action_type`: `CUT_EDGE`, `KILL_PARENT`, or `STOP`
   - `receiver`: which currently-active node to focus on
   - `parent`: which incoming edge of that receiver to act on (ignored if `STOP`)

2. **Validate**. Mask out invalid choices (already-cut edges, dead nodes). If the agent picks `STOP`, jump to terminal scoring.

3. **Apply the action**:
   - `CUT_EDGE`: mark one specific edge as cut.
   - `KILL_PARENT`: mark all outgoing edges of the parent component as cut everywhere in the graph (a macro action — useful when the agent is confident a whole head/MLP is dead weight).

4. **Run the model with the new mask**. The ablation engine installs hooks: for every cut edge, replace the activation flowing through it with the corresponding cached corrupted activation. Get new logits.

5. **Compute per-step reward**:
   - Compare new logits to previous logits → "did this cut hurt the answer?"
   - Add a small bonus for every edge gone (sparsity pressure).
   - If the cut barely moved the output: small positive reward (free shrinkage).
   - If the cut broke the answer: negative reward.

6. **Update bookkeeping**: edge mask, last-logits, active-node list (some nodes may now be orphaned), step counter.

7. **Return** (observation, reward, terminated, truncated, info).

### Episode end

Two ways to end:

- **Agent says STOP** (intended). Compute terminal reward:
  ```
  faithfulness = -KL(full_logits ‖ current_logits)   # or logit-diff for templated tasks
  minimality   = 1 - (n_active_edges / n_candidate_edges)
  terminal_reward = faithfulness × minimality
  ```
- **Step budget exceeded** (truncated, default 500 steps). Same terminal calculation, possibly with a small penalty for not stopping voluntarily — we want the agent to learn when enough is enough.

Log the final subgraph. Done.

---

## Why pre-filter with attribution patching

Without pre-filtering, the agent faces ~32,000 edges to consider. Most of them are noise on any given task — a copy-suppression task uses maybe 50 edges meaningfully; the other ~31,950 don't matter at all.

Attribution patching estimates, in three model passes, which edges would actually move the answer if cut. Throwing away the dead-obvious ones up front lets the agent spend its decision budget on the edges that actually require judgment.

**Why not just use attribution patching as the full solution?** It's a first-order linear estimate. It misses OR gates (two edges that each look weak alone but together carry the signal) and negative heads (edges that hurt and should be cut). The RL agent's job is to handle those — the cases where naive gradient methods fail.

**Why keep 3,000 and not 1,000?** Buffer against the OR-gate problem. An edge involved in an OR gate has a small individual score; we want enough headroom to keep some of those in the candidate set.

**Validation gate before trusting the pre-filter**: on IOI, check that the top-3k retains ≥80% of Wang et al.'s 26 canonical heads' edges. If not, raise the cutoff. **Do not move past this check.**

---

## The factored action space

A flat softmax over 3,000 edges is wide and hard to learn. We split the decision in two:

- **Pick a receiver node** (~200 active nodes typically): "which node do I want to mess with?"
- **Pick an incoming edge of that receiver** (~5–50 options): "given I'm focused here, which sender is the suspect?"

Same final action (a specific edge to cut), reached in two smaller decisions. Each head is smaller, gets more gradient per option, and the two picks correspond to genuinely different reasoning steps (structural vs. local).

The third action component, `action_type`, lets the agent choose:
- **Fine** (`CUT_EDGE`): one edge at a time.
- **Coarse** (`KILL_PARENT`): kill the whole upstream component's outputs.
- **Done** (`STOP`): declare the current subgraph the answer.

This gives the expressiveness of hierarchical RL (the agent can be coarse when confident) without the brittleness of a hard two-phase split where early mistakes are unrecoverable.

---

## Observation space

What the agent sees each step:

- **Edge mask**: boolean over all ~3,000 candidate edges (cut vs. alive).
- **Node features** (per active node, ~8–16 floats each):
  - layer index, head index (or MLP flag), node-type one-hot
  - current output norm
  - sum of attribution scores of incoming candidate edges
  - count of currently-active incoming and outgoing edges
- **Edge features** (per candidate edge, ~4 floats each):
  - prefilter attribution score
  - currently cut or alive
  - layer distance between endpoints
- **Task id**: one-hot over the configured task pool.
- **Step counter**: how far into the episode.

The policy is a transformer that attends over active nodes (≤1k, usually far fewer as the graph shrinks), not over all 32k edges. Decoding an edge happens conditional on the chosen receiver — same factoring as the action space.

---

## Reward design

Two components, both important:

**Per-step reward (dense, every step)**:
```
step_reward = -ΔKL(previous_logits, current_logits) + sparsity_bonus_per_edge_cut
```
- Dense signal so the agent can learn over a 500-step horizon (sparse reward on this horizon will not learn in our compute budget).
- The sparsity bonus is small but pushes the agent to actually shrink the graph.

**Terminal reward (once, at STOP)**:
```
terminal_reward = faithfulness × minimality
```
- For templated tasks, faithfulness uses logit-diff against the task's target tokens.
- For OpenWebText (Stage 4), faithfulness uses KL divergence against the full model's next-token distribution.

**Ablation method: corrupted activations, not zero ablation.** Zhang & Nanda (2023) show zero ablation gives misleading signals — knocking activations to zero pushes the model far off-manifold. We use corrupted prompts (same-structure, different-content) as the source of replacement activations, following Wang et al.'s ABC-patching methodology.

**Validation gate before trusting the reward** (Stage 2 of the plan): the canonical IOI subgraph (Wang et al.'s 26 heads) must score significantly higher under this reward than random subgraphs of the same size, across all five tasks. If not, the reward is broken. Do not start RL training until this passes.

---

## Code layout

```
mechrl/
├── env/
│   ├── graph.py              # GraphState: wraps TLACDCCorrespondence + mutable mask
│   ├── prefilter.py          # attribution_prefilter() → top-k candidate edges
│   ├── ablation.py           # AblationEngine: corrupted cache, hook installation
│   ├── reward.py             # per-step and terminal reward functions
│   ├── circuit_env.py        # CircuitEnv (gym.Env): the env itself
│   └── tasks/
│       ├── base.py           # Task interface (sample_batch, metric)
│       ├── ioi.py
│       ├── greaterthan.py
│       ├── induction.py
│       ├── docstring.py
│       └── copysuppression.py
└── tests/
    └── test_env_smoke.py
```

Dependencies on the existing ACDC code (in `paperCodes/Automatic-Circuit-Discovery/acdc/`):
- `TLACDCCorrespondence` — the DAG. We use as-is.
- `TLACDCEdge` (Edge, EdgeType, TorchIndex) — edge primitives. Use as-is.
- `TLACDCExperiment` — runs the model with masked edges. Subclass and wrap with a cleaner mask-based API.

---

## Build order and validation gates

Build incrementally. Stop at each gate.

1. **`tasks/ioi.py`** first. Best-documented task, easiest to verify against published results.
2. **`AblationEngine`**, wrapping `TLACDCExperiment`. Sanity checks:
   - Cut nothing → logits match the full model exactly.
   - Cut everything → logits match the corrupted-prompt model.
3. **`Prefilter`**. Sanity check: top-3k on IOI retains ≥80% of Wang et al.'s 26 canonical heads' edges. **Gate: do not proceed if this fails.**
4. **`RewardFn`**. Run the Stage 2 validation experiment: canonical IOI subgraph scores significantly higher than random subgraphs of matched size. **Gate: do not start RL training until this passes for all five tasks.**
5. **`CircuitEnv`** end-to-end. Smoke test with a random policy: 1,000 episodes, no crashes, faithfulness scores in a plausible range.
6. Add the other four tasks one by one, smoke-testing each.

Why incremental: ablation logic has many silent failure modes (wrong hook order, stale cache, off-by-one in indices). Building the whole stack then debugging end-to-end loses days.

---

## Key design choices and rationale

**Wrap ACDC's DAG, don't rebuild.** Their `TLACDCCorrespondence` and edge-cut machinery are correct and battle-tested. Reimplementing risks silent ablation bugs that break faithfulness scoring — exactly the failure mode that kills Stage 2.

**Attribution pre-filter + factored discrete actions** (chosen over flat actions, hard hierarchical RL, or continuous masks):
- Pre-filter collapses 32k → 3k cheaply (3 model passes per episode).
- Factored actions make the remaining 3k learnable (each policy head is small).
- Macro `KILL_PARENT` action gives hierarchical expressiveness without the brittleness of a two-phase split.
- Continuous masks were rejected: credit assignment over 32k continuous dials is a research project on its own, and "global nudge vector" doesn't transfer cleanly between tasks.

**Corrupted activations, not zero ablation.** Zero ablation gives misleading signals (Zhang & Nanda 2023). Corrupted activations follow Wang et al.'s ABC-patching and stay on-manifold.

**Dense per-step reward + terminal reward.** Sparse terminal-only reward won't learn over a 500-step horizon with our compute budget.

**STOP as an action, not a fixed threshold.** ACDC tunes a threshold to decide minimality. We let the agent learn when to stop — that's what makes minimality learnable rather than hand-tuned.

---

## Open questions / things to revisit

- **Pre-filter cadence**: re-run attribution patching every N steps as the graph shrinks? Newly-important edges might surface. Decide based on Stage 2 ablations.
- **Step budget**: 500 is a starting guess. Tune based on observed episode lengths in early training.
- **OR-gate handling**: if pre-filter loses critical OR-gate edges even at top-3k, may need a second pass that scores edges *given* the candidate set is committed.
- **Cross-episode memory**: belongs to the policy, not the env, but the env needs to expose stable node/edge IDs across episodes for the memory to attach to anything meaningful.

---

## One-paragraph summary

Pick a task. Generate prompts. Pre-filter the graph from 32k edges to ~3k candidates using one quick gradient pass. Show the agent the candidate graph. Agent picks a node it wants to look at, picks an edge into that node to cut (or a whole upstream component to kill). Environment runs the model with that cut, sees how much the output moved, hands back a small reward. Repeat until the agent says STOP or 500 moves pass. Score the final subgraph against the full model. Log. Next episode, possibly a different task, but the policy keeps its memory.
