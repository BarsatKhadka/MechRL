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
from mechrl.agent import CircuitPolicy
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
    raise ValueError(
        f"Unknown --tasks {spec!r}. Use one of {list(TASK_SETS)} or a class name in {list(_BY_NAME)}."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="all", help="task set or single class name")
    p.add_argument("--num-examples", type=int, default=None, help="override per-task example count")
    p.add_argument("--k", type=int, default=3000, help="top-K candidate edges")
    p.add_argument("--step-budget", type=int, default=400)
    p.add_argument("--sparsity-weight", type=float, default=0.001)
    p.add_argument("--invalid-penalty", type=float, default=-0.01)
    # PPO
    p.add_argument("--total-iterations", type=int, default=500)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=128)
    # infra
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available; falling back to cpu")
        device = "cpu"
    torch.manual_seed(args.seed)

    classes = resolve_tasks(args.tasks)
    run_name = f"{args.tasks}_seed{args.seed}_{int(time.time())}"
    out = Path(args.out) / run_name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "config.json").open("w") as f:
        json.dump(vars(args) | {"device": device, "task_classes": [c.__name__ for c in classes]}, f, indent=2)
    print(f"[run] {run_name}  device={device}  tasks={[c.__name__ for c in classes]}", flush=True)

    # ---- build bundles (per-task EAP-IG prefilter, cached) ----
    bundles = []
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

    # ---- env + policy + PPO ----
    env = CircuitEnv(
        bundles,
        step_budget=args.step_budget,
        sparsity_weight=args.sparsity_weight,
        invalid_penalty=args.invalid_penalty,
        seed=args.seed,
    )
    policy = CircuitPolicy(hidden=args.hidden)
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
    )
    print(f"\n[done] checkpoints + metrics in {out}", flush=True)


if __name__ == "__main__":
    main()
