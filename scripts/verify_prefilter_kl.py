"""Baseline diagnostic: how much KL-faith can the candidate set reach, per task?

The top-K candidate set is the CEILING — the agent can never beat it. If a task's
ceiling is low, that task is broken before training starts (this was docstring:
~0.51-0.64). The ceiling has two levers:
    * WHICH edges  -> attribution metric: logit-diff (legacy) vs KL (new)
    * HOW MANY      -> K

This sweeps both, per task, and prints the candidate-set KL-faithfulness so we can
see exactly what fixes each low task: better selection (KL column beats logit-diff),
more edges (faith rises with K), or neither (inherently diffuse -> drop / accept).

Run on a GPU (attribution = forward+backward x ig_steps, per task per metric):
    python -m scripts.verify_prefilter_kl --tasks all,CopySuppressionTask,SuccessorHeadsTask,InductionTask \
        --ks 1500,3000,5000,8000 --num-examples 20 --device cuda

(`all` = the 13 training tasks; add held-out class names to cover them too.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechrl.env import AblationEngine, Prefilter, build_graph
from mechrl.env.shared_model import build_shared_gpt2, use_shared_gpt2
from mechrl.train.train_agent import resolve_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help="comma-list of set/class names (see train_agent)")
    p.add_argument("--ks", default="1500,3000,5000,8000", help="comma-list of K values to sweep")
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument("--ig-steps", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="ignore cached scores, recompute")
    p.add_argument("--out", default=None, help="if set, write faith-vs-K per task/attr to this JSON")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available; using cpu", flush=True)
        device = "cpu"

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    classes = resolve_tasks(args.tasks)
    shared = build_shared_gpt2(device)

    kcols = "".join(f"K={k:<6d}" for k in ks)
    header = f"{'task':28s} {'attr':9s} {kcols}"
    print("\n" + header)
    print("-" * len(header))

    low = []     # best ceiling at the largest K still < 0.85
    failed = []  # task didn't even build/score
    data = {"ks": ks, "tasks": {}}   # tasks[name][attr] = {K: faith}  for the figure
    with use_shared_gpt2(shared):
        for cls in classes:
            try:
                task = cls(num_examples=args.num_examples, device=device)
                graph = build_graph(task.model)
                engine = AblationEngine(task, graph, metric_type="kl")

                best_at_maxk = -1.0
                for mt, label in ((None, "logitdiff"), ("kl", "kl-attr")):
                    pref = Prefilter(task, graph, ig_steps=args.ig_steps, metric_type=mt)
                    pref.compute(force=args.force)               # writes scores into `graph`
                    cells = ""
                    for k in ks:
                        faith = engine.faithfulness(pref.candidate_mask(k))   # read before next compute
                        cells += f"{faith:<8.3f}"
                        data["tasks"].setdefault(cls.__name__, {}).setdefault(label, {})[k] = float(faith)
                        if k == ks[-1]:
                            best_at_maxk = max(best_at_maxk, faith)
                    print(f"{cls.__name__:28s} {label:9s} {cells}", flush=True)
                if best_at_maxk < 0.85:
                    low.append((cls.__name__, best_at_maxk))
            except Exception as e:  # one bad task shouldn't kill the whole sweep
                failed.append((cls.__name__, repr(e)))
                print(f"{cls.__name__:28s} FAILED   {type(e).__name__}: {e}", flush=True)
            print("-" * len(header))

    if low:
        print("\nSTILL LOW (best ceiling < 0.85 even at max K) -> drop from set or accept lower:")
        for name, f in low:
            print(f"  {name:28s} {f:.3f}")
    else:
        print("\nAll built tasks reach >=0.85 ceiling at some K/attribution. Pick the cheapest "
              "(smallest K, and kl only if it beats logitdiff) per task for the real runs.")
    if failed:
        print("\nFAILED TO BUILD/SCORE (investigate separately):")
        for name, err in failed:
            print(f"  {name:28s} {err}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"\n[prefilter] faith-vs-K saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
