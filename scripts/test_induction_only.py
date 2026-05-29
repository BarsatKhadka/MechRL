"""Just induction — old (logit-diff) vs new (task's log-P) prefilter.

Olsson et al. measure induction via probability of the correct next-token
at the repeated position. Our task's metric does exactly this (log P(correct)
at the final position). The OLD prefilter used logit-diff against a random
distractor — wrong objective for induction.

Quick test: compute faithfulness at top-K with both prefilters, compare.
"""

from mechrl.tasks import InductionTask
from mechrl.env import build_graph, Prefilter, AblationEngine


def main():
    print("Loading induction task...")
    task = InductionTask(num_examples=20, half_len=25, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)

    # Reference points (compute once, reused)
    full = engine.full_baseline()
    cut = engine.corrupted_baseline()
    print(f"\nFull baseline:    {full:+.4f}")
    print(f"Cut baseline:     {cut:+.4f}")
    print(f"Dynamic range:    {full - cut:+.4f}")

    print(f"\n{'K':>6} | {'OLD (logit-diff)':>16} | {'NEW (task log-P)':>16}")
    print(f"{'-'*6} | {'-'*16} | {'-'*16}")

    # Compute OLD prefilter once
    print("\nComputing OLD prefilter (logit-diff)...")
    pref_old = Prefilter(task, graph, ig_steps=5, use_task_metric=False)
    pref_old.compute(batch_size=10)

    # Compute NEW prefilter once
    print("Computing NEW prefilter (task's natural metric)...")
    pref_new = Prefilter(task, graph, ig_steps=5, use_task_metric=True)
    pref_new.compute(batch_size=10)

    # Compare top-K faithfulness for several K values
    for k in [500, 1000, 3000, 5000, 10000]:
        f_old = engine.faithfulness(pref_old.candidate_mask(k))
        f_new = engine.faithfulness(pref_new.candidate_mask(k))
        print(f"{k:>6} | {f_old:>15.2%}  | {f_new:>15.2%}")


if __name__ == "__main__":
    main()
