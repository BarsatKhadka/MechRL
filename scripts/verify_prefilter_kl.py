"""Does KL-attribution lift the candidate-set faith CEILING on diffuse tasks?

The prefilter picks the top-K candidate edges the agent is allowed to keep. If that
set can't reproduce the full model's distribution, the agent's KL-faith is hard-capped
no matter how well it searches. On IOI/greater-than the logit-diff candidate set is
already great (~0.95+); on docstring it caps low (~0.5-0.65) because logit-diff-important
edges != distribution-reproducing edges.

This script measures, per task, the KL-faithfulness of the FULL top-K candidate set
(the ceiling) under:
    * logit-diff attribution  (legacy: metric_type=None)
    * KL attribution          (new:    metric_type="kl")
A lift on docstring (and no regression on IOI/greater-than) means the fix works and
should be used for the multi-task / held-out runs.

Run on a GPU cluster (attribution = forward+backward x ig_steps per task):
    python -m scripts.verify_prefilter_kl \
        --tasks IOITask,GreaterThanOriginal,DocstringGPT2Task,DocstringGPT2Google5Task,DocstringGPT2Sphinx7Task \
        --num-examples 20 --device cuda
"""

from __future__ import annotations

import argparse

import torch

from mechrl.env import AblationEngine, Prefilter, build_graph
from mechrl.env.shared_model import build_shared_gpt2, use_shared_gpt2
from mechrl.train.train_agent import resolve_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help="comma-list of set/class names (see train_agent)")
    p.add_argument("--k", type=int, default=3000)
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument("--ig-steps", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="ignore cached scores, recompute")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available; using cpu", flush=True)
        device = "cpu"

    classes = resolve_tasks(args.tasks)
    shared = build_shared_gpt2(device)

    print(f"\n{'task':30s} {'logit-diff':>12s} {'KL-attr':>12s} {'lift':>8s}   verdict")
    print("-" * 78)
    rows = []
    with use_shared_gpt2(shared):
        for cls in classes:
            task = cls(num_examples=args.num_examples, device=device)
            graph = build_graph(task.model)
            engine = AblationEngine(task, graph, metric_type="kl")

            ceil = {}
            for mt in (None, "kl"):
                pref = Prefilter(task, graph, ig_steps=args.ig_steps, metric_type=mt)
                pref.compute(force=args.force)          # writes scores into `graph`
                mask = pref.candidate_mask(args.k)      # read BEFORE next compute overwrites
                ceil[mt] = engine.faithfulness(mask)

            ld, kl = ceil[None], ceil["kl"]
            lift = kl - ld
            verdict = "✓ lift" if lift > 0.02 else ("~ same" if abs(lift) <= 0.02 else "✗ worse")
            print(f"{cls.__name__:30s} {ld:12.4f} {kl:12.4f} {lift:+8.4f}   {verdict}", flush=True)
            rows.append((cls.__name__, ld, kl))

    print("-" * 78)
    print("Use --prefilter-metric kl for multi-task IF docstring lifts toward >=0.85 and "
          "IOI/greater-than don't regress. If docstring still caps low, raise --k or drop "
          "it from the held-out set (use copy-suppression / successor-heads instead).")


if __name__ == "__main__":
    main()
