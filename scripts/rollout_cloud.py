"""Log the policy's rollout exploration as a (size, faithfulness) point cloud per behaviour.

For a design-space-exploration figure (cf. SwiftCTS Fig 5): every deployment runs 17 rollouts
of ~150 pruning steps, so the policy visits ~2.5k intermediate circuits per behaviour -- the
dense "explored" cloud -- and returns the best-of-sixteen -- the tight "returned" cluster that
lands in the small-and-faithful corner of its own search. We log (size, faith) at every step.

Run where the ckpt + prefilter live (GPU compute node, or the laptop -- NOT a cluster login node):
    python -m scripts.rollout_cloud --run runs/train_seed1_<ts> --device cuda \
        --tasks IOITask,GreaterThanOriginal,DocstringGPT2Task,GenderedPronounTask,AcronymTask,\
SubjectVerbAgreementTask,SimpleSyllogismTask,OppositeSyllogismTask,CountryCapitalTask,MCQAnchoredBiasTask \
        --num-rollouts 12 --out runs/rollout_cloud.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechrl.env import CircuitEnv, TaskBundle
from scripts.battery_test import TASKS, move_obs
from scripts.interaction_edges import load_policy


def explore(policy, cfg, task_name, device, num_rollouts):
    tkwargs = {"device": device}
    if cfg.get("num_examples") is not None:
        tkwargs["num_examples"] = cfg["num_examples"]
    task = TASKS[task_name](**tkwargs)
    pm = "kl" if task_name == "MCQAnchoredBiasTask" else None
    bundle = TaskBundle.build(task, k=cfg.get("k", 3000), prefilter_metric=pm)
    engine = bundle.engine
    env = CircuitEnv([bundle], step_budget=cfg.get("step_budget", 150),
                     faith_threshold=cfg.get("faith_threshold", 0.8),
                     threshold_penalty=cfg.get("threshold_penalty", 5.0),
                     invalid_penalty=cfg.get("invalid_penalty", -0.01),
                     seed=cfg.get("seed", 0), faith_margin=cfg.get("faith_margin"))

    def rollout(greedy):
        obs = env.reset(bundle_idx=0)
        traj = []
        with torch.no_grad():
            while not env.done:
                a, _, _, _ = policy.act(move_obs(obs, device), greedy=greedy)
                obs, _, _, _ = env.step(a)
                m = env.mask.clone().cpu()
                traj.append((int(m.sum().item()), float(engine.faithfulness(m))))
        return traj

    explored, returned, best = [], [], None
    for r in range(num_rollouts + 1):                  # rollout 0 = greedy, rest sampled
        traj = rollout(greedy=(r == 0))
        explored.extend(traj)
        if traj:
            sz, f = traj[-1]
            returned.append([sz, f])
            if best is None or f > best[1]:
                best = [sz, f]
    print(f"  {task_name}: {len(explored)} explored, {len(returned)} returned, "
          f"best |C|={best[0]} f={best[1]:.3f}", flush=True)
    return {"task": task_name, "explored": explored, "returned": returned, "best": best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="donor run dir (the trained policy)")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-rollouts", type=int, default=12)
    p.add_argument("--tasks", required=True, help="comma-separated task class names")
    p.add_argument("--out", default="runs/rollout_cloud.json")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable; cpu", flush=True); device = "cpu"
    policy, cfg, ckpt = load_policy(Path(args.run), args.ckpt, device)
    print(f"[cloud] donor={Path(args.run).name} ckpt={ckpt.name} device={device}", flush=True)

    rows = []
    for t in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        if t not in TASKS:
            print(f"[skip] unknown task {t}", flush=True); continue
        rows.append(explore(policy, cfg, t, device, args.num_rollouts))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=2, default=float))
    print(f"\n[cloud] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
