# MechRL Tasks — Reference

Single source of truth for which tasks we use, which we dropped, and **why**. Every
decision here is backed by a measured number from `scripts/probe_task_ceiling.py`
(not a guess). If you're tempted to re-add a dropped task or add a new one, read the
**inclusion gate** and **construction lessons** first.

Last verified: 2026-06-08 (Magnolia L40, `--num-examples 20`, logit-diff attribution).

---

## TL;DR — the locked roster

**TRAIN — 3 families, 12 tasks** (all have a sparse circuit at K=3000):

| family | mechanism | tasks | top-3K faith |
|---|---|---|---|
| IOI | name movement | IOITask, IOIAfterOpener, IOINoPlaceObject, IOIFriendsFound | 0.92–0.99 |
| GreaterThan | numeric comparison | GreaterThanOriginal, Reversed, BeganEnded, TookPlace | 0.95–0.99 |
| Docstring | next-arg (induction-ish) | DocstringGPT2Task (base), class_sphinx, sphinx_desc, func_sphinx | 0.85–0.92 |

**HELD-OUT — 3 *distinct* mechanisms** (the transfer test; not trained on):

| task | mechanism | top-3K faith | KL_cut |
|---|---|---|---|
| CopySuppressionTask | copy-suppression | 0.993 | 4.65 |
| GenderedPronounTask | coreference | 0.996 | 3.73 |
| SubjectVerbAgreementTask | syntax | 0.958 | 0.82 ⚠️ |

Within-family transfer option: also hold out **IOIFriendsFound** (or one docstring
variant) from TRAIN to measure warm-start-vs-scratch on a *same-family* task.

`--tasks` example for the train set:
`IOITask,IOIAfterOpener,IOINoPlaceObject,IOIFriendsFound,greaterthan,DocstringGPT2Task` (+ the docstring variant classes).

---

## What makes a usable task — the inclusion gate

The agent prunes the **top-K candidate edge set**; it can never beat that set's
faithfulness. So a task is only usable if the candidate set can REPRODUCE the full
model. Two numbers decide it:

- **`KL_cut`** = `KL(full-model ‖ all-corrupted)` — how much the corrupted
  counterfactual changes the output. This is the denominator of faith
  (`faith = 1 − KL(circuit)/KL_cut`). **Healthy ≈ 3–5** (IOI 4.65). **< ~1.5 = weak
  counterfactual** (faith hypersensitive). **≈ 0 = the model doesn't do the task** (no
  behavior to find a circuit for).
- **`faith@3000`** = faithfulness of the top-3000 candidate set = the **ceiling**. Must
  reach ≈1.0 at K=ALL (sanity); the question is whether it's high at K=3000.

**GATE: keep a task iff `faith@3000 ≥ 0.85` AND `KL_cut` is healthy (not ~0).**

Two failure modes (opposite fixes — don't confuse them):
- **Weak counterfactual** → `KL_cut` small. The corrupted prompt doesn't break the
  behavior. Fix = a counterfactual that flips the answer.
- **Diffuse circuit** → `KL_cut` healthy but `faith@3000` low and only reaches ~1.0 as
  K→all edges. No *sparse* circuit exists. Fix = in-distribution data, or drop.

---

## Training bar (τ) — derived from the ceiling, NOT a uniform guess

The reward penalizes faith below τ. A uniform τ is wrong here: docstring's ceiling
(~0.88) is *below* IOI's τ (0.95), so a uniform 0.95 would make docstring **permanently
unclearable**. Instead, each task's bar is derived from its own measured ceiling:

> **`τ_task = clamp(ceiling_task − margin, [0.50, 0.98])`**, via `--faith-margin` (e.g. 0.05).

The ceiling is measured automatically at `TaskBundle.build` (`engine.faithfulness(candidate_mask)`,
one forward) and printed at startup as `[bar] <Task> ceiling=.. -> tau=..`. So IOI gets
~0.90, GreaterThan ~0.93, Docstring ~0.83, GenderedPronoun ~0.95 — each clearable and
task-appropriate. Per-task logging marks ✓ when a task clears *its own* τ and records
`first_success_iter` (the forward-transfer signal). Uniform `--faith-threshold` still
exists for single-task reproduction (e.g. the locked IOI run at 0.95).

**Ceiling ≠ agent-achievable.** The table below is the *candidate-set* ceiling (best
possible). Whether the *agent* reaches near it is a separate check — confirmed for IOI
(~0.94); verify others single-task (`slurm/train_single.sbatch`) before multi-task.

---

## Full ceiling table (everything we measured)

`faith@K` = top-K candidate-set KL-faith. `—` = not separately measured (faith suffices).

| task | family | KL_cut | faith@3k | faith@8k | status | note |
|---|---|---|---|---|---|---|
| IOITask | IOI | 4.65 | 0.951 | 0.981 | ✅ TRAIN | control |
| IOIAfterOpener | IOI | — | 0.949 | 0.961 | ✅ TRAIN | |
| IOINoPlaceObject | IOI | — | 0.991 | 0.992 | ✅ TRAIN | |
| IOIFriendsFound | IOI | — | 0.927 | 0.964 | ✅ TRAIN / held-out option | |
| GreaterThanOriginal | GT | — | 0.980 | 0.983 | ✅ TRAIN | |
| GreaterThanReversed | GT | — | 0.954 | 0.980 | ✅ TRAIN | |
| GreaterThanBeganEnded | GT | — | 0.975 | 0.974 | ✅ TRAIN | |
| GreaterThanTookPlace | GT | — | 0.983 | 0.988 | ✅ TRAIN | |
| DocstringGPT2Task (sphinx_5) | Docstring | 2.45 | 0.918 | 0.956 | ✅ TRAIN | base, `:param` cue |
| DocstringGPT2ClassSphinxTask | Docstring | 2.18 | 0.846 | 0.928 | ✅ TRAIN | n=50 |
| DocstringSphinxDescTask (new) | Docstring | 2.37 | 0.847 | 0.918 | ✅ TRAIN | `:param`+descriptions |
| DocstringFuncSphinxTask (new) | Docstring | 2.06 | 0.865 | 0.918 | ✅ TRAIN | free function |
| CopySuppressionTask | copy-suppr | 4.65 | 0.993 | 0.995 | ✅ HELD-OUT | head 10.7 |
| GenderedPronounTask (new) | coreference | 3.73 | 0.996 | 0.998 | ✅ HELD-OUT | top-tier |
| SubjectVerbAgreementTask (new) | syntax | 0.82 | 0.958 | 0.985 | ✅ HELD-OUT ⚠️ | sparse+real but small KL_cut headroom |
| DocstringArgFieldTask (new) | Docstring | 1.32 | 0.813 | 0.874 | ❌ drop | borderline, `:arg` weaker than `:param` |
| DocstringGPT2Sphinx7Task | Docstring | 1.05 | 0.635 | 0.814 | ❌ drop | 7 args dilutes "next arg" |
| DocstringGPT2Google5Task | Docstring | 0.61 | 0.511 | 0.727 | ❌ drop | ends in whitespace, no field cue |
| DocstringGPT2Numpy5Task | Docstring | — | 0.725 | 0.786 | ❌ drop | ends in whitespace |
| SuccessorHeadsTask | successor | 0.02 | 0.409 | 0.679 | ❌ drop | GPT-2-small doesn't do it (see below) |
| InductionTask | induction | 5.93 | 0.317 | 0.661 | ❌ drop | diffuse (see below) |

---

## Dropped tasks — with evidence (so we don't re-litigate)

**Induction — DIFFUSE, not fixable.** `KL_cut` is healthy (5.9), so the counterfactual
is fine, but `faith@3000=0.317` and it only reaches 0.90 at **K=16000** (~half the
graph). GPT-2 processes random-token sequences (OOD) with many components → no sparse
circuit. Tested fixes, both **worse**: longer sequences (hl=8→0.317, hl=40→0.199) and
contiguous real text (`real_text=True`, faith@3k 0.018–0.14). Canonical induction lives
in `redwood_attn_2l`, a different model — it does not transfer cleanly to GPT-2 under
full-distribution KL. (Code kept: `InductionTask(real_text=...)`, but unused.)

**Successor — WEAK BEHAVIOR, not fixable.** `KL_cut ≈ 0.02` even with a single category
(months 0.011 / days 0.007 / numbers 0.061). That means GPT-2-small's output barely
changes between "The day after Monday is" and "…Wednesday is" — **it isn't doing
successor**. (My padding-artifact hypothesis was wrong; single-category removed the EOS
padding and KL_cut stayed ~0 → it's the model, not the prompt.) Successor heads are
prominent in larger models, not GPT-2-small. (Code kept: `SuccessorHeadsTask(only_category=...)`.)

**Weak docstring variants (google5 / numpy5 / sphinx7 / arg_field).** All end in a weak
final cue → low `KL_cut` → low ceiling. See lesson 2 below.

---

## Construction lessons — READ before adding/fixing a task

1. **Single-token slots + fixed template ⇒ uniform length ⇒ NO EOS padding.** Mixing
   variable-length prompts forces left/right EOS padding, which round-trips through
   `to_string`→re-tokenize and **swamps the signal** (this is what gave successor
   `KL_cut≈0`). Filter every fill-in slot to a single GPT-2 token and assert uniform
   length (see `subject_verb.py` / `gendered_pronoun.py`).
2. **Put a STRONG cue right before the predicted token.** Docstring `:param`/`:arg`
   (model is confident an arg name follows) → `KL_cut≈2.4`. Bare whitespace
   (google/numpy styles) → the model isn't sure anything specific comes next →
   `KL_cut≈0.6`. The cue determines how sharp the output is, hence `KL_cut`.
3. **The counterfactual must FLIP the answer.** Swap the one element that determines the
   output: IOI names, GT year, docstring signature order, SVA subject number, pronoun
   name-gender. Big output change → healthy `KL_cut`. A counterfactual that leaves the
   answer ~unchanged gives `KL_cut≈0` and a useless faith signal.
4. **In-distribution prompts ⇒ sparse circuit.** Random/OOD inputs make GPT-2 spread
   computation across many heads (induction). Real, natural prompts keep the circuit
   small. A strong `KL_cut` does NOT guarantee sparseness (induction had both strong
   `KL_cut` and a diffuse circuit).
5. **Sanity: `faith@ALL` must be ≈1.0.** Keeping all edges = the full model. If it isn't
   ~1.0, the engine/graph is mis-wired for that task, not a ceiling problem.
6. **`KL_cut` reading:** ≈0 → non-behavior (drop). <1.5 → weak counterfactual (usable but
   tight headroom; SVA at 0.82 works because its circuit is very sparse). 3–5 → healthy.

---

## How to verify a new task

```bash
# add the task to --tasks (Part 1 prints KL_cut + faith vs K)
python -m scripts.probe_task_ceiling --tasks IOITask,YourNewTask --force --device cuda
# Magnolia: edit slurm/probe_docstring.sbatch --tasks line, sbatch it.
```
Keep it iff `faith@3000 ≥ 0.85` and `KL_cut` healthy. The probe also sweeps
K∈{3000,8000,16000,ALL} so you can see whether more K rescues it (coverage) or not
(diffuse).

---

## File map

| what | where |
|---|---|
| IOI | `mechrl/tasks/ioi.py`, `ioi_variants.py` |
| GreaterThan | `mechrl/tasks/greaterthan.py`, `greaterthan_variants.py`, `greaterthan_helpers.py` |
| Docstring (GPT-2) | `mechrl/tasks/docstring_gpt2.py`, `docstring_variants.py` |
| CopySuppression | `mechrl/tasks/copy_suppression.py` |
| Subject-Verb Agreement (new) | `mechrl/tasks/subject_verb.py` |
| Gendered Pronoun (new) | `mechrl/tasks/gendered_pronoun.py` |
| Induction / Successor (DROPPED) | `mechrl/tasks/induction.py`, `successor_heads.py` |
| task registry / `--tasks` resolution | `mechrl/tasks/__init__.py`, `mechrl/train/train_agent.py` |
| ceiling probe | `scripts/probe_task_ceiling.py`, `slurm/probe_*.sbatch` |
| candidate-attribution sweep | `scripts/verify_prefilter_kl.py` |

---

## References (documented GPT-2-small circuits)

- IOI — Wang et al. 2022, [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Greater-Than — Hanna et al. 2023
- Copy Suppression (head 10.7) — McDougall et al. 2023
- Docstring — Heimersheim & Janiak 2023 (orig. on attn-only-4l; adapted to GPT-2 here)
- Subject-Verb Agreement / Verb Conjugation — [arXiv:2506.22105](https://arxiv.org/abs/2506.22105) (12-head circuit), [Finlayson 2021](https://arxiv.org/abs/2106.06087)
- Gendered Pronoun — Mathwin et al. (MI hackathon) / ACDC
- Acronyms (not built — multi-token) — [García-Carrasco 2024, arXiv:2405.04156](https://arxiv.org/abs/2405.04156)
- Induction (canonical on redwood_attn_2l, NOT GPT-2) — Olsson et al. 2022
- Successor Heads (prominent in larger models) — Gould et al. 2024, [arXiv:2312.09230](https://arxiv.org/abs/2312.09230)
