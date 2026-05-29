"""Recompute top-3000 faithfulness for the 3 training tasks.

The agent will start each episode at these faithfulness levels and try to
PRUNE the graph while staying close to this starting faithfulness. The
transfer claim is about pruning EFFICIENCY (fewer passes), not about
exceeding these starting numbers.
"""

from __future__ import annotations

import torch

from mechrl.tasks import IOITask, GreaterThanTask, DocstringGPT2Task
from mechrl.env import build_graph, Prefilter, AblationEngine


TRAINING_TASKS = [
    ("IOI", IOITask, {"num_examples": 30}),
    ("greater-than", GreaterThanTask, {"num_examples": 20}),
    ("docstring-gpt2", DocstringGPT2Task, {"num_examples": 20}),
]


def main():
    print(f"\n{'Training task':>18} | {'full':>8} | {'cut':>8} | "
          f"{'top-1000':>9} | {'top-2000':>9} | {'top-3000':>9}")
    print(f"{'-'*18} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*9}")

    for name, cls, kwargs in TRAINING_TASKS:
        task = cls(device="cpu", **kwargs)
        graph = build_graph(task.model)
        engine = AblationEngine(task, graph)
        pref = Prefilter(task, graph, ig_steps=5)
        pref.compute(batch_size=10)

        full = engine.full_baseline()
        cut = engine.corrupted_baseline()

        f_1k = engine.faithfulness(pref.candidate_mask(1000))
        f_2k = engine.faithfulness(pref.candidate_mask(2000))
        f_3k = engine.faithfulness(pref.candidate_mask(3000))

        print(f"{name:>18} | {full:>+8.3f} | {cut:>+8.3f} | "
              f"{f_1k:>8.2%} | {f_2k:>8.2%} | {f_3k:>8.2%}")


if __name__ == "__main__":
    main()
