"""Consolidated pre-policy verification over ALL training tasks.

For each of the 13 training task classes:
  - builds the bundle (cached prefilter + engine + features)
  - Gate 1 proxy: full vs cut baselines are well-separated (model does the task)
  - Gate 2: top-3k faithfulness > 60% (circuit is inside the candidate set)
  - Feature sanity: signed score in [-1,1], rank in [0,1], some helper edges

Prints one row per task and a final PASS/FAIL summary. Designed to run
sequentially (one model in memory at a time) to stay light on the machine.
"""

import gc
import sys

import torch

from mechrl.tasks import TRAINING_TASK_CLASSES
from mechrl.env import TaskBundle


def main():
    hdr = f"{'task':>26} | {'full':>8} | {'cut':>8} | {'gap':>7} | {'top-3k':>8} | {'G2':>4} | {'feat':>5}"
    print(hdr)
    print("-" * len(hdr))
    sys.stdout.flush()

    results = []
    for cls in TRAINING_TASK_CLASSES:
        name = cls.__name__
        try:
            task = cls(device="cpu")
            bundle = TaskBundle.build(task, k=3000)
            engine = bundle.engine

            full = engine.full_baseline()
            cut = engine.corrupted_baseline()
            gap = full - cut
            faith = engine.faithfulness(bundle.candidate_mask)

            # feature sanity
            signed = bundle.edge_features[:, 0]
            rank = bundle.edge_features[:, 1]
            feat_ok = (
                signed.abs().max().item() <= 1.0 + 1e-5
                and 0.0 - 1e-5 <= rank.min().item()
                and rank.max().item() <= 1.0 + 1e-5
                and bool((signed < 0).any())
            )

            g2 = faith > 0.60
            results.append((name, g2, feat_ok))
            print(f"{name:>26} | {full:>+8.3f} | {cut:>+8.3f} | {gap:>+7.3f} | "
                  f"{faith:>7.2%} | {'OK' if g2 else 'LOW':>4} | {'OK' if feat_ok else 'BAD':>5}")
            sys.stdout.flush()

            del task, bundle, engine
            gc.collect()
        except Exception as e:
            msg = str(e).encode("ascii", "replace").decode("ascii")[:60]
            results.append((name, False, False))
            print(f"{name:>26} | ERROR: {msg}")
            sys.stdout.flush()

    print("\n=== summary ===")
    n_g2 = sum(1 for _, g2, _ in results if g2)
    n_feat = sum(1 for _, _, f in results if f)
    print(f"Gate 2 (top-3k faith > 60%):  {n_g2}/{len(results)} tasks")
    print(f"Feature sanity:               {n_feat}/{len(results)} tasks")
    bad = [n for n, g2, f in results if not (g2 and f)]
    if bad:
        print(f"NEEDS ATTENTION: {bad}")
    else:
        print("ALL TRAINING TASKS GREEN.")


if __name__ == "__main__":
    main()
