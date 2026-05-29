"""Run the ablation engine on all GPT-2 small tasks.

For each task, runs the 3 sanity tests:
  Test 1: run_with_mask(all_alive) should equal full_baseline (proves correctness)
  Test 2: run_with_mask(all_cut)   should be much lower / different
  Test 3: run_with_mask(50% random) — informative variability check

Note: AblationEngine currently uses logit-diff as its metric (good for IOI,
docstring-style tasks). For tasks where logit-diff isn't the natural metric
(e.g. greater-than uses prob-diff, copy-suppression uses raw logit), the
numbers are less interpretable but the correctness check (Test 1 exact match)
still validates the ablation machinery.
"""

from __future__ import annotations

import torch

from mechrl.tasks import (
    IOITask,
    GreaterThanTask,
    InductionTask,
    DocstringGPT2Task,
    CopySuppressionTask,
    SuccessorHeadsTask,
)
from mechrl.env import build_graph, AblationEngine


TASKS = [
    ("ioi", IOITask, {"num_examples": 30}),
    ("greaterthan", GreaterThanTask, {"num_examples": 20}),
    ("induction", InductionTask, {"num_examples": 20, "half_len": 25}),
    ("docstring_gpt2", DocstringGPT2Task, {"num_examples": 20}),
    ("copy_suppression", CopySuppressionTask, {"num_examples": 20}),
    ("successor_heads", SuccessorHeadsTask, {"num_examples": 20}),
]


def test_task(name, cls, kwargs):
    print(f"\n{'='*70}\nTASK: {name}\n{'='*70}")

    task = cls(device="cpu", **kwargs)
    model = task.model
    graph = build_graph(model)
    engine = AblationEngine(task, graph)  # default batch_size = task.num_examples

    print(f"  Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"  Engine: {engine.n_edges} edges total")

    full = engine.full_baseline()
    print(f"\n  Full baseline (no ablation):  {full:+.4f}")

    print("\n  Test 1: all alive == full baseline?")
    score_alive = engine.run_with_mask(engine.all_alive_mask())
    diff1 = abs(score_alive - full)
    status1 = "PASS" if diff1 < 0.01 else "FAIL"
    print(f"    all_alive: {score_alive:+.4f}   diff: {diff1:.6f}   [{status1}]")

    print("\n  Test 2: all cut should be different from full baseline")
    score_cut = engine.run_with_mask(engine.all_cut_mask())
    diff2 = abs(score_cut - full)
    status2 = "PASS" if diff2 > 0.1 else "WARN"
    print(f"    all_cut:   {score_cut:+.4f}   diff from full: {diff2:.4f}   [{status2}]")

    print("\n  Test 3: 50% random cut — informative variability")
    torch.manual_seed(0)
    random_mask = torch.rand(engine.n_edges) < 0.5
    score_random = engine.run_with_mask(random_mask)
    print(f"    50% random: {score_random:+.4f}")
    print(f"    (full {full:+.3f}, all_cut {score_cut:+.3f})")

    return {
        "name": name,
        "full": full,
        "all_alive": score_alive,
        "all_cut": score_cut,
        "random_50": score_random,
        "test1_pass": diff1 < 0.01,
        "test2_pass": diff2 > 0.1,
    }


def main():
    results = []
    for name, cls, kwargs in TASKS:
        try:
            r = test_task(name, cls, kwargs)
            results.append(r)
        except Exception as e:
            msg = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"\n  FAILED on {name}: {type(e).__name__}: {msg}")
            results.append({"name": name, "error": msg})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'task':>20} | {'full':>8} | {'all_alive':>8} | {'all_cut':>8} | {'random':>8} | Test1 | Test2")
    print(f"{'-'*20} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*5} | {'-'*5}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:>20} | ERROR: {r['error'][:60]}")
            continue
        p1 = "PASS" if r["test1_pass"] else "FAIL"
        p2 = "PASS" if r["test2_pass"] else "WARN"
        print(f"{r['name']:>20} | {r['full']:>+8.3f} | {r['all_alive']:>+8.3f} | "
              f"{r['all_cut']:>+8.3f} | {r['random_50']:>+8.3f} | {p1:>5} | {p2:>5}")


if __name__ == "__main__":
    main()
