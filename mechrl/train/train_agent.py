"""Train the circuit-finding agent with PPO on one or more tasks.

Run via the module entry point (so imports resolve):
    python -m mechrl.train.train_agent --tasks all --device cuda

Task selection (--tasks):
    all          -> all 13 verified training tasks (IOI+greaterthan+docstring)
    ioi          -> the 4 IOI variants
    greaterthan  -> the 4 greater-than variants
    docstring    -> the 5 docstring variants
    <ClassName>  -> a single task class, e.g. IOITask  (for the single-task signal run)

Outputs go to <out>/<run_name>/: config.json, metrics.jsonl, policy checkpoints.
The expensive per-task setup (EAP-IG prefilter) is cached on disk; on GPU each
task builds in seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from mechrl.tasks import (
    TRAINING_TASK_CLASSES,
    IOITask, IOIAfterOpener, IOINoPlaceObject, IOIFriendsFound,
    GreaterThanOriginal, GreaterThanReversed, GreaterThanBeganEnded, GreaterThanTookPlace,
    DocstringGPT2Task, DocstringGPT2Sphinx7Task, DocstringGPT2Google5Task,
    DocstringGPT2ClassSphinxTask, DocstringGPT2Numpy5Task,
)
from mechrl.env import CircuitEnv, TaskBundle
from mechrl.env.shared_model import build_shared_gpt2, use_shared_gpt2
from mechrl.agent import CircuitPolicy
from mechrl.agent.batch_policy import BatchCutPolicy
from mechrl.train import PPOConfig, PPOTrainer


TASK_SETS = {
    "all": TRAINING_TASK_CLASSES,
    "ioi": [IOITask, IOIAfterOpener, IOINoPlaceObject, IOIFriendsFound],
    "greaterthan": [GreaterThanOriginal, GreaterThanReversed,
                    GreaterThanBeganEnded, GreaterThanTookPlace],
    "docstring": [DocstringGPT2Task, DocstringGPT2Sphinx7Task, DocstringGPT2Google5Task,
                  DocstringGPT2ClassSphinxTask, DocstringGPT2Numpy5Task],
}
_BY_NAME = {c.__name__: c for c in TRAINING_TASK_CLASSES}


def resolve_tasks(spec: str):
    if spec in TASK_SETS:
        return TASK_SETS[spec]
    if spec in _BY_NAME:
        return [_BY_NAME[spec]]
    # comma-list of set names and/or class names, e.g. "ioi,greaterthan" (8 tasks)
    # or "IOITask,GreaterThanOriginal". Lets you compose a multi-task TRAIN set and
    # keep a separate held-out set for the transfer eval. Order preserved, deduped.
    if "," in spec:
        out, seen = [], set()
        for part in (p.strip() for p in spec.split(",") if p.strip()):
            group = TASK_SETS.get(part) or ([_BY_NAME[part]] if part in _BY_NAME else None)
            if group is None:
                raise ValueError(f"Unknown task {part!r} in --tasks {spec!r}")
            for c in group:
                if c not in seen:
                    seen.add(c); out.append(c)
        return out
    raise ValueError(
        f"Unknown --tasks {spec!r}. Use one of {list(TASK_SETS)}, a class name in "
        f"{list(_BY_NAME)}, or a comma-list of those."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="all", help="task set or single class name")
    p.add_argument("--num-examples", type=int, default=None, help="override per-task example count")
    p.add_argument("--k", type=int, default=3000, help="top-K candidate edges")
    p.add_argument("--step-budget", type=int, default=400)
    p.add_argument("--faith-threshold", type=float, default=0.8, help="tau: faithfulness bar to clear")
    p.add_argument("--threshold-penalty", type=float, default=3.0, help="lambda: how hard the threshold is")
    p.add_argument("--minimality-weight", type=float, default=1.0,
                   help="w: how strongly cutting edges is rewarded vs the faith penalty. "
                        "Raise above 1.0 if the agent is too risk-averse (keeps too many edges).")
    p.add_argument("--invalid-penalty", type=float, default=-0.01)
    # PPO
    p.add_argument("--total-iterations", type=int, default=500)
    # num-steps should be >= step-budget so each rollout finishes >=1 episode
    # (otherwise per-iter episode metrics are nan on rollouts that complete none).
    p.add_argument("--num-steps", type=int, default=512)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--policy", choices=["flat", "batch"], default="flat",
                   help="flat = legacy single-cut/kill/stop; batch = autoregressive batch-cut")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 3, 10, 30, 100],
                   help="batch-cut size options (only used with --policy batch)")
    # infra
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--resume-from", default=None,
                   help="path to a run dir; reload its latest policy_iter*.pt and continue")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available; falling back to cpu")
        device = "cpu"
    torch.manual_seed(args.seed)

    classes = resolve_tasks(args.tasks)

    # ---- resume vs fresh run ----
    start_iter = 0
    resume_ckpt = None
    if args.resume_from is not None:
        out = Path(args.resume_from)
        ckpts = sorted(out.glob("policy_iter*.pt"),
                       key=lambda p: int(re.search(r"iter(\d+)", p.name).group(1)))
        if not ckpts:
            raise FileNotFoundError(f"no policy_iter*.pt in {out}")
        resume_ckpt = ckpts[-1]
        start_iter = int(re.search(r"iter(\d+)", resume_ckpt.name).group(1))
        print(f"[resume] {out.name}  from {resume_ckpt.name} (iter {start_iter})", flush=True)
    else:
        run_name = f"{args.tasks}_seed{args.seed}_{int(time.time())}"
        out = Path(args.out) / run_name
        out.mkdir(parents=True, exist_ok=True)
        with (out / "config.json").open("w") as f:
            json.dump(vars(args) | {"device": device, "task_classes": [c.__name__ for c in classes]}, f, indent=2)
        print(f"[run] {run_name}  device={device}  tasks={[c.__name__ for c in classes]}", flush=True)

    # ---- build bundles (per-task EAP-IG prefilter, cached) ----
    # ONE frozen GPT-2 shared across all tasks (N copies would OOM the 8GB GPU).
    # Identical model per task, so single-task numerics/resumes are unchanged.
    shared_model = build_shared_gpt2(device)
    bundles = []
    with use_shared_gpt2(shared_model):
        for cls in classes:
            kwargs = {"device": device}
            if args.num_examples is not None:
                kwargs["num_examples"] = args.num_examples
            t0 = time.time()
            task = cls(**kwargs)
            bundle = TaskBundle.build(task, k=args.k)
            bundles.append(bundle)
            print(f"  built {cls.__name__:28s} K={bundle.n_candidates} "
                  f"M={len(bundle.parent_names)} ({time.time()-t0:.0f}s)", flush=True)
    assert all(b.task.model is shared_model for b in bundles), \
        "tasks did not share the model -- shared-GPT-2 interception failed"
    print(f"  [shared] 1 GPT-2 across {len(bundles)} task(s)", flush=True)

    # ---- env + policy + PPO ----
    env = CircuitEnv(
        bundles,
        step_budget=args.step_budget,
        faith_threshold=args.faith_threshold,
        threshold_penalty=args.threshold_penalty,
        invalid_penalty=args.invalid_penalty,
        seed=args.seed,
        minimality_weight=args.minimality_weight,
    )
    if args.policy == "batch":
        policy = BatchCutPolicy(hidden=args.hidden, batch_sizes=tuple(args.batch_sizes))
        print(f"[policy] BatchCutPolicy batch_sizes={args.batch_sizes}", flush=True)
    else:
        policy = CircuitPolicy(hidden=args.hidden)
    if resume_ckpt is not None:
        policy.load_state_dict(torch.load(resume_ckpt, map_location=device))
        print(f"[resume] loaded weights from {resume_ckpt.name}", flush=True)
    cfg = PPOConfig(
        total_iterations=args.total_iterations,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        seed=args.seed,
    )
    trainer = PPOTrainer(env, policy, cfg, device=device)

    print(f"[train] {args.total_iterations} iterations x {args.num_steps} steps "
          f"(budget {args.step_budget}) on {len(bundles)} task(s)\n", flush=True)
    trainer.train(
        log_every=args.log_every,
        save_dir=out,
        save_every=args.save_every,
        metrics_path=out / "metrics.jsonl",
        start_iter=start_iter,
    )
    print(f"\n[done] checkpoints + metrics in {out}", flush=True)


if __name__ == "__main__":
    main()
