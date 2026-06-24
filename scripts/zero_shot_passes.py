"""Count the EXACT forward passes a frozen (zero-shot) policy spends per behaviour.

Each env step evaluates faithfulness with one model forward pass (engine.run_with_mask),
and so does scoring each finished rollout. The inference cost of producing a circuit is
therefore the number of run_with_mask calls across the best-of-K rollouts (1 greedy + K
sampled). run_with_mask is the SAME one-forward-pass primitive that scripts.run_acdc_greedy
counts, so this number is directly comparable to ACDC's `passes` column.

We wrap run_with_mask with a counter, zero it AFTER the one-time baseline setup, then run
the same rollouts as battery_test.extract_circuit. Per task we report: total forward passes
for the full best-of-K, the passes of a single rollout (greedy), mean steps/episode, and the
chosen circuit's |C| / faith.

Run on the box with the ckpt + prefilter cache (loads GPT-2 -> compute node, NOT login):
    python -m scripts.zero_shot_passes --run runs/train_seed1_<ts> --device cuda \
        --tasks IOITask,GreaterThanOriginal,DocstringGPT2Task,GenderedPronounTask,\
SubjectVerbAgreementTask,AcronymTask,SimpleSyllogismTask,OppositeSyllogismTask,\
CountryCapitalTask,MCQAnchoredBiasTask --num-rollouts 16 --out runs/zeroshot_passes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechrl.env import CircuitEnv, TaskBundle
from mechrl.agent.batch_policy import BatchCutPolicy
from scripts.battery_test import TASKS, latest_ckpt, move_obs


def count_for_task(policy, cfg, task_name, device, num_rollouts):
    """Build the task's env, run best-of-K frozen rollouts, count run_with_mask calls."""
    tkwargs = {"device": device}
    if cfg.get("num_examples") is not None:
        tkwargs["num_examples"] = cfg["num_examples"]
    task = TASKS[task_name](**tkwargs)
    # MCQ is invisible to the logit-diff prefilter; it needs KL attribution (matches the paper).
    pm = "kl" if task_name == "MCQAnchoredBiasTask" else None
    bundle = TaskBundle.build(task, k=cfg.get("k", 3000), prefilter_metric=pm)
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

    # Wrap the one-forward-pass primitive with a call counter.
    orig = engine.run_with_mask
    state = {"calls": 0}
    def counted(*a, **kw):                   # run_with_mask(mask, intervention=...) -> pass all through
        state["calls"] += 1
        return orig(*a, **kw)
    engine.run_with_mask = counted

    def rollout(greedy):
        before = state["calls"]
        steps = 0
        obs = env.reset(bundle_idx=0)
        with torch.no_grad():
            while not env.done:
                action, _, _, _ = policy.act(move_obs(obs, device), greedy=greedy)
                obs, _, _, _ = env.step(action)
                steps += 1
        m = env.mask.clone().cpu()
        return m, engine.faithfulness(m), int(m.sum().item()), steps, state["calls"] - before

    state["calls"] = 0                       # count only inference, not one-time setup
    gm, gf, gk, gsteps, gpass = rollout(greedy=True)
    best = (gm, gf, gk)
    steps_list = [gsteps]
    for _ in range(num_rollouts):
        m, f, kp, st, _ = rollout(greedy=False)
        steps_list.append(st)
        if f > best[1]:
            best = (m, f, kp)
    total_passes = state["calls"]
    engine.run_with_mask = orig              # restore

    return {
        "task": task_name,
        "k": cfg.get("k", 3000),
        "num_rollouts": num_rollouts,
        "rollouts_total": num_rollouts + 1,          # +1 greedy
        "forward_passes_total": total_passes,        # full best-of-K inference cost
        "forward_passes_one_rollout": gpass,         # a single (greedy) rollout
        "mean_steps_per_episode": sum(steps_list) / len(steps_list),
        "max_steps_per_episode": max(steps_list),
        "final_edges": best[2],
        "final_faith": best[1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="DONOR run dir (config.json + policy_iter*.pt)")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-rollouts", type=int, default=16)
    p.add_argument("--tasks", required=True, help="comma-separated task class names")
    p.add_argument("--out", default="runs/zeroshot_passes.json")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable; cpu", flush=True)
        device = "cpu"

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(run_dir)
    print(f"[passes] donor={run_dir.name} ckpt={ckpt.name} device={device}", flush=True)

    policy = BatchCutPolicy(hidden=cfg.get("hidden", 128),
                            batch_sizes=tuple(cfg.get("batch_sizes", [1, 3, 10, 30]))).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    rows = []
    for t in tasks:
        if t not in TASKS:
            print(f"[skip] unknown task {t}", flush=True)
            continue
        print(f"\n=== {t} ===", flush=True)
        r = count_for_task(policy, cfg, t, device, args.num_rollouts)
        rows.append(r)
        print(f"  total fwd passes (best-of-{args.num_rollouts}) = {r['forward_passes_total']}  "
              f"| 1 rollout = {r['forward_passes_one_rollout']}  "
              f"| mean steps/ep = {r['mean_steps_per_episode']:.1f} (max {r['max_steps_per_episode']})  "
              f"| {r['final_edges']} edges @ f={r['final_faith']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"donor": run_dir.name, "rows": rows}, indent=2, default=float))
    print(f"\n[passes] saved -> {args.out}", flush=True)

    print(f"\n{'behaviour':26} {'total_passes':12} {'1_rollout':10} {'mean_steps':10} {'edges':6} {'faith'}")
    for r in rows:
        print(f"{r['task']:26} {r['forward_passes_total']:<12} {r['forward_passes_one_rollout']:<10} "
              f"{r['mean_steps_per_episode']:<10.1f} {r['final_edges']:<6} {r['final_faith']:.3f}")


if __name__ == "__main__":
    main()
