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
    # Garcia-Carrasco et al. 2024 (AISTATS) -- letter-movers + previous-token + propagation.
    "AcronymTask": {(8, 11), (10, 10), (9, 9), (11, 4), (4, 11), (1, 0), (2, 2), (5, 8)},
    # Saraipour & Zhang 2025 -- truth heads (Fig 2b).
    "SimpleSyllogismTask": {(7, 2), (9, 1), (9, 9), (10, 1), (10, 4)},
    # Saraipour & Zhang 2025 -- negative-truth heads.
    "OppositeSyllogismTask": {(7, 3), (8, 8), (8, 10), (9, 7), (10, 7)},
}
MLPS_BY_TASK = {
    # Opposite syllogism also relies on truth-logit-rescaler MLPs (layers 7-10).
    "OppositeSyllogismTask": {7, 8, 9, 10},
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
    bundle = TaskBundle.build(task, k=cfg.get("k", 3000))
    engine = bundle.engine

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


if __name__ == "__main__":
    main()
