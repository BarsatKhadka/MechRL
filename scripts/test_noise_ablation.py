"""Does IBCircuit-style task-agnostic ablation rescue the tasks we dropped?

For each task, print the candidate-set ceiling under THREE ablations:
  * patching  -- corrupted-prompt activations (current; needs a good counterfactual)
  * mean      -- dataset-mean activation        (task-agnostic, already in eap)
  * noise     -- Gaussian eps ~ N(mean, std^2)  (IBCircuit, arXiv:2602.22581)

The dropped tasks (successor/induction/diffuse docstrings) failed under PATCHING because
their corrupted counterfactual didn't disrupt the behaviour (KL_cut ~0). mean/noise don't
need a counterfactual -- if KL_cut becomes healthy AND faith@K stays high, the task is
rescued and the failure was a counterfactual artifact, not the model.

GPU:
    python -m scripts.test_noise_ablation \
        --tasks SuccessorHeadsTask,InductionTask,DocstringGPT2Google5Task,IOITask --device cuda
"""

from __future__ import annotations

import argparse

import torch

from mechrl.env import AblationEngine, Prefilter, build_graph
from mechrl.env.shared_model import build_shared_gpt2, use_shared_gpt2
from mechrl.train.train_agent import resolve_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--k", type=int, default=3000)
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available; using cpu", flush=True)
        device = "cpu"

    classes = resolve_tasks(args.tasks)
    shared = build_shared_gpt2(device)
    print(f"\n{'task':30s} {'ablation':10s} {'KL_cut':>8s} {'faith@'+str(args.k):>11s}")
    print("-" * 64)
    with use_shared_gpt2(shared):
        for cls in classes:
            try:
                task = cls(num_examples=args.num_examples, device=device)
                graph = build_graph(task.model)
                engine = AblationEngine(task, graph, metric_type="kl")
                pref = Prefilter(task=task, graph=graph, metric_type=None)
                pref.compute()
                mask = pref.candidate_mask(min(args.k, len(graph.edges)))
                for interv in ("patching", "mean", "noise"):
                    cut = engine.cut_baseline(interv)
                    faith = engine.faithfulness(mask, intervention=interv)
                    flag = "  <- healthy" if (cut >= 1.5 and faith >= 0.85) else ""
                    print(f"{cls.__name__:30s} {interv:10s} {cut:8.3f} {faith:11.3f}{flag}", flush=True)
                print("-" * 64)
            except Exception as e:
                print(f"{cls.__name__:30s} FAILED  {type(e).__name__}: {e}", flush=True)
                print("-" * 64)

    print("\nRescued = a DROPPED task whose mean/noise row has healthy KL_cut (>~1.5) AND")
    print("high faith@K, where patching was ~0. That means task-agnostic ablation works")
    print("and the task can come back (and answers 'why not noise like IBCircuit?').")


if __name__ == "__main__":
    main()
