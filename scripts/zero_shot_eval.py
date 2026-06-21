"""ZERO-SHOT eval: apply a FROZEN policy to a task with NO fine-tuning.

Loads a donor policy, points it at --task, samples best-of-K rollouts (no PPO, no
gradient steps), and scores the circuit it produces cold -- on the same paper metric
as the battery (GreaterThan: prob-diff; IOI: logit-diff + heads).

  - If --task was in the donor's training set -> this is a frozen-replay sanity
    (does the policy reproduce its circuit without re-training?).
  - If --task is HELD-OUT (donor never trained on it) -> this is ZERO-SHOT TRANSFER:
    does the learned cutting skill produce a faithful circuit on an unseen task with
    ZERO training? (The strongest transfer claim -- stronger than warm-start/few-shot.)

Run (on the box with the ckpt + prefilter cache):
    python -m scripts.zero_shot_eval \
        --run runs/IOITask,GreaterThanOriginal_seed0_<ts> \
        --task GreaterThanOriginal --device cuda --num-rollouts 16
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from mechrl.env import CircuitEnv, TaskBundle
from mechrl.agent.batch_policy import BatchCutPolicy
from mechrl.tasks.greaterthan_helpers import get_yy_token_ids

# Reuse the battery's frozen-rollout extractor, metric evals, task registry, ckpt finder.
from scripts.battery_test import (
    TASKS, latest_ckpt, extract_circuit, gt_evaluate, IOI_FAMILIES,
)
from scripts.evaluate_circuit import (
    evaluate_circuit as ioi_evaluate, task_accuracy, random_mask, parse_head,
)


# Canonical components from the discovery papers, for HELD-OUT head recovery (informational,
# like greater-than: the agent finds an edge-level circuit; we report how many of the paper's
# heads/MLPs it contains). (layer, head); MLPs by layer index.
HEADS_BY_TASK = {
    # Mathwin et al. 2023 (Apart Alignment Jam) -- named significant heads, by position group:
    # name (0,4)(2,6); is (4,3)(6,0); late (9,7)(10,9)(11,8). We drop L1H4, which the paper
    # lists in the name group but explicitly marks "No effect" (Sec 3.3) -- a non-head by its
    # own causal account. PRELIMINARY + MLP-heavy (the paper ignored MLPs), so informational.
    "GenderedPronounTask": {(0, 4), (2, 6), (4, 3), (6, 0), (9, 7), (10, 9), (11, 8)},
    # Garcia-Carrasco et al. 2024 (AISTATS) -- letter-movers + previous-token + propagation.
    "AcronymTask": {(8, 11), (10, 10), (9, 9), (11, 4), (4, 11), (1, 0), (2, 2), (5, 8)},
    # Saraipour & Zhang 2025 -- truth heads (Fig 2b).
    "SimpleSyllogismTask": {(7, 2), (9, 1), (9, 9), (10, 1), (10, 4)},
    # Saraipour & Zhang 2025 -- negative-truth heads.
    "OppositeSyllogismTask": {(7, 3), (8, 8), (8, 10), (9, 7), (10, 7)},
    # Li & Gao 2025 -- anchored-bias heads (+ MLP9 below).
    "MCQAnchoredBiasTask": {(8, 1), (10, 8)},
    # arXiv:2506.22105 "Identifying a Circuit for Verb Conjugation in GPT-2" -- 12-head base
    # circuit (no MLPs): subject-anchor + scanner + conjunction-tracker + invariant.
    "SubjectVerbAgreementTask": {(0, 4), (0, 8), (1, 0), (1, 1), (2, 1), (2, 6),
                                 (6, 0), (9, 4), (10, 0), (11, 4), (11, 6), (11, 7)},
    # arXiv:2403.19521 "Interpreting Key Mechanisms of Factual Recall" -- argument-passers
    # (9,8)(10,0), negative (10,7)(11,10), positive (11,1)(11,5)(11,6)(11,9). Informational
    # (some are general factual-recall machinery, not country-capital specific).
    "CountryCapitalTask": {(9, 8), (10, 0), (10, 7), (11, 10), (11, 1), (11, 5), (11, 6), (11, 9)},
}
MLPS_BY_TASK = {
    # Opposite syllogism also relies on truth-logit-rescaler MLPs (layers 7-10).
    "OppositeSyllogismTask": {7, 8, 9, 10},
    "MCQAnchoredBiasTask": {9},
    "CountryCapitalTask": {9, 10, 11},
}


def head_recovery(engine, mask, heads_canon, mlps_canon=None):
    """Scan the circuit's edges for the paper's canonical heads/MLPs (informational)."""
    heads, mlps = set(), set()
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        e = engine.edge_list[i]
        for nm in (e.parent.name, e.child.name):
            ph = parse_head(nm)
            if ph is not None:
                heads.add(ph)
            mm = re.fullmatch(r"m(?:lp)?(\d+)", nm)
            if mm is not None:
                mlps.add(int(mm.group(1)))
    rec = {"heads_recovered": sorted(heads_canon & heads), "heads_total": len(heads_canon)}
    if mlps_canon:
        rec["mlps_recovered"] = sorted(mlps_canon & mlps)
        rec["mlps_total"] = len(mlps_canon)
    return rec


def generic_evaluate(engine, mask):
    """Held-out logit-diff tasks (GenderedPronoun he/she, SVA is/are, CopySuppression):
    faith + argmax/logit-diff recovery (no task-specific ground-truth heads) +
    necessity + specificity. Works for any task whose dataset carries (correct,wrong) labels."""
    n = int(mask.sum().item())
    kl = engine.run_with_mask(mask)
    kl_cut = engine.corrupted_baseline()
    faith = engine.faithfulness(mask)
    acc = task_accuracy(engine, mask)                      # argmax (+ logit-diff if labels present)
    knock = engine.all_alive_mask(); knock[mask] = False
    knockout_faith = engine.faithfulness(knock)
    rand_faith = engine.faithfulness(random_mask(engine.n_edges, n, seed=0))
    return {"n_edges": n, "kl": kl, "kl_cut": kl_cut, "faith": faith, **acc,
            "knockout_faith": knockout_faith, "random_same_size_faith": rand_faith}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="DONOR run dir (config.json + policy_iter*.pt)")
    p.add_argument("--task", default="GreaterThanOriginal", help=f"target task: {list(TASKS)}")
    p.add_argument("--ckpt", default=None, help="checkpoint (default latest; e.g. policy_iter600.pt)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-rollouts", type=int, default=16, help="best-of-K frozen rollouts")
    p.add_argument("--prefilter-metric", choices=["task", "kl"], default=None,
                   help="prefilter for the candidate set. None=logit-diff (donor default); "
                        "'kl' = KL attribution, matches the KL faithfulness objective and is "
                        "REQUIRED for tasks logit-diff misses (e.g. MCQAnchoredBiasTask, whose "
                        "circuit is invisible under logit-diff but reaches ~0.95 faith under kl).")
    p.add_argument("--dump-circuit", action="store_true",
                   help="also write circuit_<task>.json with EVERY edge and head in the returned "
                        "circuit, the canonical head/MLP set, and the intersection, so head "
                        "recovery can be verified by hand rather than trusted.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available; using cpu", flush=True)
        device = "cpu"
    if args.task not in TASKS:
        raise ValueError(f"--task must be one of {list(TASKS)}")

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(run_dir)
    trained_on = cfg.get("task_classes", [])
    held_out = args.task not in trained_on
    print(f"[zero-shot] donor={run_dir.name} ckpt={ckpt.name}", flush=True)
    print(f"[zero-shot] target={args.task}  donor_trained_on={trained_on}", flush=True)
    print(f"[zero-shot] => {'HELD-OUT (true zero-shot transfer)' if held_out else 'in training set (frozen-replay sanity)'}", flush=True)

    # Build the TARGET task (may differ from the donor's training tasks).
    tkwargs = {"device": device}
    if cfg.get("num_examples") is not None:
        tkwargs["num_examples"] = cfg["num_examples"]
    task = TASKS[args.task](**tkwargs)
    pm = "kl" if args.prefilter_metric == "kl" else None
    bundle = TaskBundle.build(task, k=cfg.get("k", 3000), prefilter_metric=pm)
    engine = bundle.engine
    if pm == "kl":
        print(f"[zero-shot] prefilter = KL attribution (overriding donor default)", flush=True)

    env = CircuitEnv(
        [bundle],
        step_budget=cfg.get("step_budget", 150),
        faith_threshold=cfg.get("faith_threshold", 0.8),
        threshold_penalty=cfg.get("threshold_penalty", 5.0),
        invalid_penalty=cfg.get("invalid_penalty", -0.01),
        seed=cfg.get("seed", 0),
        faith_margin=cfg.get("faith_margin"),
    )
    print(f"  [bar] {args.task} ceiling={bundle.ceiling:.3f} -> tau={env.taus[0]:.3f}", flush=True)

    # FROZEN policy -- architecture from the donor's config, weights from the ckpt.
    policy = BatchCutPolicy(hidden=cfg.get("hidden", 128),
                            batch_sizes=tuple(cfg.get("batch_sizes", [1, 3, 10, 30]))).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    # NO training -- just frozen rollouts.
    mask = extract_circuit(policy, env, device, args.num_rollouts, args.task)

    print(f"\n=== ZERO-SHOT ({args.task}{' / HELD-OUT' if held_out else ''}) ===", flush=True)
    if args.task == "IOITask":
        res = ioi_evaluate(engine, mask)
        print(f"circuit: {res['n_edges']} edges  faith {res['faith']:.4f}", flush=True)
        print(f"[heads] canonical: {res['heads_recovered']}/{res['heads_total']}", flush=True)
        print(f"[behavior] argmax {res['argmax_accuracy']:.3f}  logit-diff recovery {res.get('logit_diff_recovery', float('nan')):.3f}  correct>wrong {res.get('correct_gt_wrong_frac', float('nan')):.3f}", flush=True)
    elif args.task == "GreaterThanOriginal":
        years_YY = task._validation.correct_labels
        yy_ids = torch.tensor(get_yy_token_ids(task.model.tokenizer))
        res = gt_evaluate(engine, mask, years_YY, yy_ids)
        print(f"circuit: {res['n_edges']} edges  faith {res['faith']:.4f}", flush=True)
        print(f"[metric: prob-diff] full {res['full_prob_diff']:+.4f}  circuit {res['circuit_prob_diff']:+.4f}  recovery {res['prob_diff_recovery']:.3f}", flush=True)
        print(f"[behavior] predicts valid year: circuit {res['circuit_valid_year_acc']:.3f}  (full {res['full_valid_year_acc']:.3f})", flush=True)
        print(f"[specificity] random same-size prob-diff {res['random_same_size_prob_diff']:+.4f}", flush=True)
    else:
        # held-out logit-diff task (GenderedPronoun / SVA / CopySuppression)
        res = generic_evaluate(engine, mask)
        print(f"circuit: {res['n_edges']} edges  faith {res['faith']:.4f}  (KL {res['kl']:.4f} / cut {res['kl_cut']:.4f})", flush=True)
        print(f"[behavior] argmax agreement {res['argmax_accuracy']:.3f}", flush=True)
        if "logit_diff_recovery" in res:
            print(f"           logit-diff recovery {res['logit_diff_recovery']:.3f}  correct>wrong {res['correct_gt_wrong_frac']:.3f}", flush=True)
        print(f"[necessity]   knockout faith {res['knockout_faith']:.4f}", flush=True)
        print(f"[specificity] random same-size faith {res['random_same_size_faith']:.4f}", flush=True)
        if args.task in HEADS_BY_TASK:
            rec = head_recovery(engine, mask, HEADS_BY_TASK[args.task], MLPS_BY_TASK.get(args.task))
            res.update(rec)
            line = f"[components]  canonical heads {len(rec['heads_recovered'])}/{rec['heads_total']} {rec['heads_recovered']}"
            if "mlps_recovered" in rec:
                line += f"  MLPs {len(rec['mlps_recovered'])}/{rec['mlps_total']} {rec['mlps_recovered']}"
            print(line + "  (informational)", flush=True)

    res["task"] = args.task
    res["donor"] = run_dir.name
    res["held_out"] = held_out
    res["mode"] = "zero_shot_frozen"
    out = Path(args.out) if args.out else (run_dir / f"zeroshot_{args.task}.json")
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[zero-shot] saved -> {out}", flush=True)

    if args.dump_circuit:
        edges, heads_present, mlps_present = [], set(), set()
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            e = engine.edge_list[i]
            edges.append(f"{e.parent.name} -> {e.child.name}" + (f" [{e.qkv}]" if e.qkv else ""))
            for nm in (e.parent.name, e.child.name):
                ph = parse_head(nm)
                if ph is not None:
                    heads_present.add(ph)
                mm = re.fullmatch(r"m(?:lp)?(\d+)", nm)
                if mm is not None:
                    mlps_present.add(int(mm.group(1)))
        canon = set(HEADS_BY_TASK.get(args.task, set()))
        canon_mlp = set(MLPS_BY_TASK.get(args.task, set()))
        dump = {
            "task": args.task,
            "faith": res.get("faith"),
            "n_edges": int(mask.sum().item()),
            "canonical_heads": sorted([list(h) for h in canon]),
            "canonical_mlps": sorted(canon_mlp),
            "heads_present": sorted([list(h) for h in heads_present]),
            "mlps_present": sorted(mlps_present),
            "heads_recovered": sorted([list(h) for h in (canon & heads_present)]),
            "mlps_recovered": sorted(canon_mlp & mlps_present),
            "edges": edges,
        }
        dpath = out.parent / f"circuit_{args.task}.json"
        dpath.write_text(json.dumps(dump, indent=2))
        print(f"[dump] circuit -> {dpath}  heads_present={len(heads_present)} "
              f"canonical={len(canon)} recovered={len(canon & heads_present)}", flush=True)


if __name__ == "__main__":
    main()
