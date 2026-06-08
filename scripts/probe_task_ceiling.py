"""Diagnose WHY a task's candidate-set ceiling is low -- two failure modes, opposite fixes.

  * WEAK COUNTERFACTUAL: KL_cut (all-corrupted vs full model) is SMALL, so faith
    (=1-KL/KL_cut) is hypersensitive and hard to push high even with a good circuit.
    Compare to IOI's KL_cut ~4.65. Fix = a counterfactual that actually destroys the
    behaviour.
  * DIFFUSE CIRCUIT: KL_cut is healthy, but faith only reaches ~1.0 as K -> all edges
    (no SMALL faithful subset exists). Fix = in-distribution data so a sparse circuit
    forms, or drop the task.

Part 1 (always): per task, full KL (~0 sanity), KL_cut, and faith at K in
    {3000, 8000, 16000, ALL} -- the shape tells diffuse-vs-counterfactual.
Part 2 (--induction-sweep): rebuild InductionTask across half_len/num_examples to see
    if a longer (more in-distribution) induction sequence gives GPT-2 a sparse circuit.

GPU:
    python -m scripts.probe_task_ceiling \
        --tasks InductionTask,SuccessorHeadsTask,DocstringGPT2Google5Task,IOITask,CopySuppressionTask \
        --induction-sweep --device cuda
"""

from __future__ import annotations

import argparse

import torch

from mechrl.env import AblationEngine, Prefilter, build_graph
from mechrl.env.shared_model import build_shared_gpt2, use_shared_gpt2
from mechrl.tasks import InductionTask, SuccessorHeadsTask
from mechrl.train.train_agent import resolve_tasks


def _probe_one(label, task, ks, force):
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph, metric_type="kl")
    _row(label, engine, graph, ks, force=force)


def _row(name, engine, graph, ks, force=False):
    full = engine.full_baseline()            # KL at all-alive -> ~0 (sanity)
    klcut = engine.corrupted_baseline()      # KL_cut: counterfactual strength
    pref = Prefilter(task=engine.task, graph=graph, metric_type=None)  # logit-diff (best here)
    pref.compute(force=force)
    n_edges = len(graph.edges)
    cells = ""
    for k in ks:
        kk = min(k, n_edges)
        cells += f"{engine.faithfulness(pref.candidate_mask(kk)):<8.3f}"
    tag = "  <-- KL_cut LOW (weak counterfactual)" if klcut < 1.5 else ""
    print(f"{name:30s} fullKL={full:6.3f}  KL_cut={klcut:6.3f}   {cells}{tag}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument("--induction-sweep", action="store_true")
    p.add_argument("--force", action="store_true", help="ignore cached prefilter scores")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available; using cpu", flush=True)
        device = "cpu"

    classes = resolve_tasks(args.tasks)
    shared = build_shared_gpt2(device)
    ks = [3000, 8000, 16000, 10 ** 9]   # last = ALL edges (capped to n_edges)

    hdr = f"{'task':30s} {'fullKL':>12s} {'KL_cut':>9s}    " + "".join(f"K={k:<6}" for k in [3000, 8000, 16000, 'ALL'])
    print("\n=== Part 1: diagnosis (KL_cut + faith vs K) ===")
    print(hdr); print("-" * len(hdr))
    with use_shared_gpt2(shared):
        for cls in classes:
            try:
                task = cls(num_examples=args.num_examples, device=device)
                graph = build_graph(task.model)
                engine = AblationEngine(task, graph, metric_type="kl")
                _row(cls.__name__, engine, graph, ks, force=args.force)
            except Exception as e:
                print(f"{cls.__name__:30s} FAILED  {type(e).__name__}: {e}", flush=True)

        if args.induction_sweep:
            print("\n=== Part 2: induction dataset sweep (does a longer seq give GPT-2 a sparse circuit?) ===")
            print(f"{'config':30s} {'fullKL':>12s} {'KL_cut':>9s}    " + "".join(f"K={k:<6}" for k in [3000, 8000, 16000, 'ALL']))
            print("-" * len(hdr))
            for half_len in (8, 16, 25, 40):
                for nex in (args.num_examples, max(40, args.num_examples)):
                    if nex != args.num_examples and half_len not in (25,):   # only sweep nex at hl=25 (keep grid small)
                        continue
                    try:
                        task = InductionTask(num_examples=nex, half_len=half_len, device=device)
                        graph = build_graph(task.model)
                        engine = AblationEngine(task, graph, metric_type="kl")
                        _row(f"Induction hl={half_len} n={nex} seq={2*half_len-1}", engine, graph, ks, force=args.force)
                    except Exception as e:
                        print(f"Induction hl={half_len} n={nex:<4} FAILED  {type(e).__name__}: {e}", flush=True)

        # --- Fix experiments (the actual point of this run) ---
        print("\n=== Part 3: FIX experiments ===")
        print(f"{'config':30s} {'fullKL':>12s} {'KL_cut':>9s}    " + "".join(f"K={k:<6}" for k in [3000, 8000, 16000, 'ALL']))
        print("-" * len(hdr))
        # Successor: single category -> uniform length, NO EOS padding (test the artifact)
        for cat in ("months", "days", "numbers"):
            try:
                _probe_one(f"Successor cat={cat}",
                           SuccessorHeadsTask(num_examples=max(args.num_examples, 40),
                                              only_category=cat, device=device), ks, args.force)
            except Exception as e:
                print(f"Successor cat={cat:<8} FAILED  {type(e).__name__}: {e}", flush=True)
        # Induction: contiguous real text (in-distribution) instead of random gibberish
        for hl in (8, 16, 25):
            try:
                _probe_one(f"Induction REALTEXT hl={hl}",
                           InductionTask(num_examples=args.num_examples, half_len=hl,
                                         real_text=True, device=device), ks, args.force)
            except Exception as e:
                print(f"Induction REALTEXT hl={hl:<3} FAILED  {type(e).__name__}: {e}", flush=True)

    print("\nRead: KL_cut<1.5 => weak counterfactual (fix the corrupted prompt). "
          "KL_cut healthy but faith low at K=3000 yet ~1.0 at K=ALL => diffuse "
          "(needs in-distribution data or drop). For induction, a higher half_len row "
          "with high K=3000 faith = the fix.")


if __name__ == "__main__":
    main()
