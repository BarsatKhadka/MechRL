"""Re-test prefilter using each task's NATURAL metric for attribution.

Comparing:
  - Old: logit-diff metric (one-size-fits-all)
  - New: each task's own metric for IG attribution

For induction especially, this should fix the 15% faithfulness issue.
"""

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
    print(f"  {'task':>20} | {'top-3000 (old)':>14} | {'top-3000 (new)':>14}")
    print(f"  {'-'*20} | {'-'*14} | {'-'*14}")
    for name, cls, kwargs in TASKS:
        try:
            task = cls(device="cpu", **kwargs)
            graph = build_graph(task.model)
            engine = AblationEngine(task, graph)

            # Old: logit-diff metric
            pref_old = Prefilter(task, graph, ig_steps=5, use_task_metric=False)
            pref_old.compute(batch_size=10)
            f_old = engine.faithfulness(pref_old.candidate_mask(3000))

            # New: task's own metric
            pref_new = Prefilter(task, graph, ig_steps=5, use_task_metric=True)
            pref_new.compute(batch_size=10)
            f_new = engine.faithfulness(pref_new.candidate_mask(3000))

            print(f"  {name:>20} | {f_old:>13.2%} | {f_new:>13.2%}")
        except Exception as e:
            msg = str(e).encode("ascii", errors="replace").decode("ascii")[:60]
            print(f"  {name:>20} | ERROR: {msg}")


if __name__ == "__main__":
    main()
