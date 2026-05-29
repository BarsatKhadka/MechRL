"""Verify normalized faithfulness gives consistent [0,1] readings across tasks.

For each GPT-2 task:
  - all_alive faithfulness should be ~1.0
  - all_cut faithfulness should be ~0.0
  - top-3000 faithfulness should be somewhere between (the agent's starting state)
"""

from __future__ import annotations

from mechrl.tasks import (
    IOITask, GreaterThanTask, InductionTask,
    DocstringGPT2Task, CopySuppressionTask, SuccessorHeadsTask,
)
from mechrl.env import build_graph, Prefilter, AblationEngine


TASKS = [
    ("ioi", IOITask, {"num_examples": 30}),
    ("greaterthan", GreaterThanTask, {"num_examples": 20}),
    ("induction", InductionTask, {"num_examples": 20, "half_len": 25}),
    ("docstring_gpt2", DocstringGPT2Task, {"num_examples": 20}),
    ("copy_suppression", CopySuppressionTask, {"num_examples": 20}),
    ("successor_heads", SuccessorHeadsTask, {"num_examples": 20}),
]


def main():
    print(f"  {'task':>20} | {'full':>7} | {'cut':>7} | {'all_alive':>9} | {'all_cut':>7} | {'top-3000':>8}")
    print(f"  {'-'*20} | {'-'*7} | {'-'*7} | {'-'*9} | {'-'*7} | {'-'*8}")
    for name, cls, kwargs in TASKS:
        try:
            task = cls(device="cpu", **kwargs)
            graph = build_graph(task.model)
            engine = AblationEngine(task, graph)
            pref = Prefilter(task, graph, ig_steps=5)
            pref.compute(batch_size=10)

            full = engine.full_baseline()
            cut = engine.corrupted_baseline()

            faith_alive = engine.faithfulness(engine.all_alive_mask())
            faith_cut = engine.faithfulness(engine.all_cut_mask())
            faith_3k = engine.faithfulness(pref.candidate_mask(3000))

            print(f"  {name:>20} | {full:>+7.3f} | {cut:>+7.3f} | "
                  f"{faith_alive:>8.2%} | {faith_cut:>6.2%} | {faith_3k:>7.2%}")
        except Exception as e:
            msg = str(e).encode("ascii", errors="replace").decode("ascii")[:50]
            print(f"  {name:>20} | ERROR: {msg}")


if __name__ == "__main__":
    main()
