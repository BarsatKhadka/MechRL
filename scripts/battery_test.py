"""Validity battery for the 2-task agent (IOI + GreaterThan), from one policy ckpt.

Flow (per task): the agent PLAYS first (best-of-K sampled rollout -> the circuit it
finds), THEN we score that circuit on the paper's metrics.

  IOI   (Wang et al. 2022 / ACDC):  metric = LOGIT DIFF (correct name - wrong name).
        Battery (reused from evaluate_circuit.py): 26-canonical-head recovery,
        argmax agreement, logit-diff recovery, necessity (knockout), specificity
        (random same-size).

  GREATER-THAN (Hanna et al. 2023 / ACDC):  metric = PROB DIFF
        ( sum P(year>YY) - sum P(year<=YY) ), the exact ACDC greater-than metric
        (mechrl.tasks.greaterthan_helpers.build_year_metric). Battery: prob-diff
        recovery, "predicts a valid year" accuracy (prob-diff>0 per example),
        necessity, specificity, and paper-component recovery (MLPs 0/8/9/10/11 +
        the year-mover heads). NOTE: greater-than's edge-level ground truth is
        coarser than IOI's 26 heads, so component recovery is informational.

Run on the box where the 2-task ckpt + prefilter cache live (CIC / Aquaman):
    python -m scripts.battery_test \
        --run runs/IOITask,GreaterThanOriginal_seed0_<ts> --device cuda \
        --out runs/battery_2task.json
(--ckpt policy_iter600.pt to pin the iter-600 policy; default = latest)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from mechrl.tasks import IOITask
from mechrl.tasks.greaterthan_variants import GreaterThanOriginal
from mechrl.tasks.greaterthan_helpers import get_yy_token_ids
from mechrl.tasks.gendered_pronoun import GenderedPronounTask
from mechrl.tasks.copy_suppression import CopySuppressionTask
from mechrl.tasks.subject_verb import SubjectVerbAgreementTask
from mechrl.tasks.acronyms import AcronymTask
from mechrl.tasks.syllogism import SimpleSyllogismTask, OppositeSyllogismTask
from mechrl.env import CircuitEnv, TaskBundle
from mechrl.agent.batch_policy import BatchCutPolicy

# Reuse the tested IOI battery + helpers.
from scripts.evaluate_circuit import (
    evaluate_circuit as ioi_evaluate,
    circuit_logits,
    random_mask,
    parse_head,
    IOI_FAMILIES,
)

TASKS = {"IOITask": IOITask, "GreaterThanOriginal": GreaterThanOriginal,
         # held-out tasks (for zero_shot_eval / transfer; not in 2-task training)
         "GenderedPronounTask": GenderedPronounTask,
         "CopySuppressionTask": CopySuppressionTask,
         "SubjectVerbAgreementTask": SubjectVerbAgreementTask,
         "AcronymTask": AcronymTask,
         "SimpleSyllogismTask": SimpleSyllogismTask,
         "OppositeSyllogismTask": OppositeSyllogismTask}

# Hanna, Liu, Variengien (2023) greater-than circuit -- the dominant components.
# MLP-heavy (the >-computation lives in late MLPs); plus year-mover attention heads.
# Coarser than IOI's canonical 26, so treat recovery as informational, not pass/fail.
GT_MLPS = {0, 8, 9, 10, 11}
GT_HEADS = {(0, 1), (0, 3), (0, 5), (0, 8),
            (5, 1), (5, 5), (6, 1), (6, 9),
            (7, 10), (8, 8), (8, 11), (9, 1)}


def latest_ckpt(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("policy_iter*.pt"),
                   key=lambda p: int(re.search(r"iter(\d+)", p.name).group(1)))
    if (run_dir / "policy_final.pt").exists() and not ckpts:
        return run_dir / "policy_final.pt"
    if not ckpts:
        raise FileNotFoundError(f"no policy_iter*.pt / policy_final.pt in {run_dir}")
    return ckpts[-1]


def move_obs(obs, device):
    return {k: v.to(device) for k, v in obs.items()}


def extract_circuit(policy, env, device, num_rollouts, label):
    """Agent PLAYS: best-of-K sampled rollouts; keep the most faithful final mask."""
    def rollout(greedy):
        obs = env.reset(bundle_idx=0)
        with torch.no_grad():
            while not env.done:
                action, _, _, _ = policy.act(move_obs(obs, device), greedy=greedy)
                obs, _, _, info = env.step(action)
        m = env.mask.clone().cpu()
        return m, env.bundle.engine.faithfulness(m), int(m.sum().item())

    # Include greedy AS A CANDIDATE: on the training tasks greedy is degenerate and a
    # sample wins; on some HELD-OUT tasks (e.g. CopySuppression) the reverse holds --
    # greedy is faithful while sampling is noisy. Take the best faith of {greedy} u {samples}.
    gm, gf, gk = rollout(greedy=True)
    print(f"  [{label}] greedy: kept {gk} faith {gf:+.3f}", flush=True)
    best, best_src = (gm, gf, gk), "greedy"
    for k in range(num_rollouts):
        m, f, kp = rollout(greedy=False)
        if f > best[1]:
            best, best_src = (m, f, kp), f"sample{k}"
    print(f"  [{label}] best: kept {best[2]} faith {best[1]:.4f}  (from {best_src})", flush=True)
    return best[0]


# ---- GreaterThan prob-diff (ACDC metric) ----

def _prob_diff_per_example(last_logits, years_YY, yy_ids):
    """sum P(year>YY) - sum P(year<=YY), per example. Positive = prefers valid years."""
    probs = F.softmax(last_logits, dim=-1)
    yy_probs = probs[:, yy_ids]                                  # [bs, 100]
    idx = torch.arange(100, device=yy_probs.device).unsqueeze(0)
    good = idx > years_YY.to(yy_probs.device).unsqueeze(1)       # [bs, 100]
    good_p = (yy_probs * good).sum(-1)
    bad_p = (yy_probs * (~good)).sum(-1)
    return good_p - bad_p


def gt_evaluate(engine, mask, years_YY, yy_ids):
    dev = engine._full_logits.device
    years_YY = years_YY.to(dev)
    yy_ids = yy_ids.to(dev)
    n = int(mask.sum().item())

    kl = engine.run_with_mask(mask)
    kl_cut = engine.corrupted_baseline()
    faith = engine.faithfulness(mask)

    # prob-diff recovery: full vs circuit (and a random same-size baseline)
    full_pd = _prob_diff_per_example(engine._full_logits[:, -1, :], years_YY, yy_ids)
    circ_logits, _ = circuit_logits(engine, mask)
    circ_pd = _prob_diff_per_example(circ_logits[:, -1, :], years_YY, yy_ids)
    full_mean, circ_mean = full_pd.mean().item(), circ_pd.mean().item()
    recovery = circ_mean / full_mean if abs(full_mean) > 1e-9 else float("nan")
    valid_frac = (circ_pd > 0).float().mean().item()
    full_valid_frac = (full_pd > 0).float().mean().item()

    rmask = random_mask(engine.n_edges, n, seed=0)
    rand_faith = engine.faithfulness(rmask)
    rand_logits, _ = circuit_logits(engine, rmask)
    rand_pd = _prob_diff_per_example(rand_logits[:, -1, :], years_YY, yy_ids).mean().item()

    # necessity: knock the circuit out of the FULL model -> faith should drop
    knock = engine.all_alive_mask()
    knock[mask] = False
    knockout_faith = engine.faithfulness(knock)

    # paper-component recovery (informational): scan circuit node names for MLPs + heads
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
    return {
        "n_edges": n,
        "kl": kl, "kl_cut": kl_cut, "faith": faith,
        "full_prob_diff": full_mean, "circuit_prob_diff": circ_mean,
        "prob_diff_recovery": recovery,
        "circuit_valid_year_acc": valid_frac, "full_valid_year_acc": full_valid_frac,
        "knockout_faith": knockout_faith,
        "random_same_size_faith": rand_faith, "random_same_size_prob_diff": rand_pd,
        "mlps_recovered": sorted(GT_MLPS & mlps), "mlps_total": len(GT_MLPS),
        "heads_recovered": sorted(GT_HEADS & heads), "heads_total": len(GT_HEADS),
    }


# ---- per-task driver ----

def run_task(task_name, cfg, ckpt, device, num_rollouts):
    print(f"\n################  {task_name}  ################", flush=True)
    tkwargs = {"device": device}
    if cfg.get("num_examples") is not None:
        tkwargs["num_examples"] = cfg["num_examples"]
    task = TASKS[task_name](**tkwargs)
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
    policy = BatchCutPolicy(hidden=cfg.get("hidden", 128),
                            batch_sizes=tuple(cfg.get("batch_sizes", [1, 3, 10, 30]))).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    mask = extract_circuit(policy, env, device, num_rollouts, task_name)

    if task_name == "IOITask":
        res = ioi_evaluate(engine, mask)
        print(f"\n=== IOI BATTERY ===  circuit: {res['n_edges']} edges", flush=True)
        print(f"[LEAD] canonical heads: {res['heads_recovered']}/{res['heads_total']}", flush=True)
        for fam, hs in IOI_FAMILIES.items():
            print(f"    {fam:20s} {res['heads_by_family'][fam]}/{len(hs)}", flush=True)
        print(f"[STRICT] argmax agreement: {res['argmax_accuracy']:.3f}", flush=True)
        if "logit_diff_recovery" in res:
            print(f"         logit-diff recovery: {res['logit_diff_recovery']:.3f}", flush=True)
            print(f"         correct>wrong frac:  {res['correct_gt_wrong_frac']:.3f}", flush=True)
        print(f"[divergence]  faith {res['faith']:.4f}  (KL {res['kl']:.4f} / cut {res['kl_cut']:.4f})", flush=True)
        print(f"[necessity]   knockout faith: {res['knockout_faith']:.4f}  (lower = more necessary)", flush=True)
        print(f"[specificity] random same-size: faith {res['random_same_size_faith']:.4f}  argmax {res['random_same_size_argmax_acc']:.3f}", flush=True)
    else:
        years_YY = task._validation.correct_labels
        yy_ids = torch.tensor(get_yy_token_ids(task.model.tokenizer))
        res = gt_evaluate(engine, mask, years_YY, yy_ids)
        print(f"\n=== GREATER-THAN BATTERY ===  circuit: {res['n_edges']} edges", flush=True)
        print(f"[metric: prob-diff]  full {res['full_prob_diff']:+.4f}  circuit {res['circuit_prob_diff']:+.4f}", flush=True)
        print(f"         prob-diff recovery: {res['prob_diff_recovery']:.3f}", flush=True)
        print(f"[behavior] predicts valid year: circuit {res['circuit_valid_year_acc']:.3f}  (full {res['full_valid_year_acc']:.3f})", flush=True)
        print(f"[divergence]  faith {res['faith']:.4f}  (KL {res['kl']:.4f} / cut {res['kl_cut']:.4f})", flush=True)
        print(f"[necessity]   knockout faith: {res['knockout_faith']:.4f}  (lower = more necessary)", flush=True)
        print(f"[specificity] random same-size: faith {res['random_same_size_faith']:.4f}  prob-diff {res['random_same_size_prob_diff']:+.4f}", flush=True)
        print(f"[components]  MLPs {res['mlps_recovered']}/{res['mlps_total']}  heads {len(res['heads_recovered'])}/{res['heads_total']} (informational)", flush=True)

    res["task"] = task_name
    res["n_edges_circuit"] = int(mask.sum().item())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="2-task run dir (config.json + policy_iter*.pt)")
    p.add_argument("--ckpt", default=None, help="checkpoint (default: latest; pass policy_iter600.pt to pin)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-rollouts", type=int, default=8, help="best-of-K sampled rollouts per task")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available; using cpu", flush=True)
        device = "cpu"

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(run_dir)
    task_names = cfg.get("task_classes", [])
    print(f"[battery] run={run_dir.name} ckpt={ckpt.name} tasks={task_names} device={device}", flush=True)

    results = {}
    for tn in task_names:
        if tn not in TASKS:
            print(f"[skip] {tn} not supported by this battery", flush=True)
            continue
        results[tn] = run_task(tn, cfg, ckpt, device, args.num_rollouts)

    out = Path(args.out) if args.out else (run_dir / "battery_2task.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[battery] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
