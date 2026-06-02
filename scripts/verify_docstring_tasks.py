"""Light verification of the 5 docstring training tasks.

Pinned to num_examples=20 so the prefilter HITS CACHE (no heavy recompute /
memory spike). One model in memory at a time, gc between tasks.
"""

import gc
import sys

from mechrl.tasks import (
    DocstringGPT2Task,
    DocstringGPT2Sphinx7Task,
    DocstringGPT2Google5Task,
    DocstringGPT2ClassSphinxTask,
    DocstringGPT2Numpy5Task,
)
from mechrl.env import TaskBundle

CLASSES = [
    DocstringGPT2Task,
    DocstringGPT2Sphinx7Task,
    DocstringGPT2Google5Task,
    DocstringGPT2ClassSphinxTask,
    DocstringGPT2Numpy5Task,
]


def main():
    hdr = f"{'task':>28} | {'full':>8} | {'cut':>8} | {'top-3k':>8} | {'G2':>4} | {'feat':>5}"
    print(hdr); print("-" * len(hdr)); sys.stdout.flush()
    results = []
    for cls in CLASSES:
        name = cls.__name__
        try:
            task = cls(num_examples=20, device="cpu")
            bundle = TaskBundle.build(task, k=3000)
            eng = bundle.engine
            full = eng.full_baseline(); cut = eng.corrupted_baseline()
            faith = eng.faithfulness(bundle.candidate_mask)
            signed = bundle.edge_features[:, 0]; rank = bundle.edge_features[:, 1]
            feat_ok = (signed.abs().max().item() <= 1.0 + 1e-5
                       and rank.max().item() <= 1.0 + 1e-5
                       and bool((signed < 0).any()))
            g2 = faith > 0.60
            results.append((name, g2, feat_ok))
            print(f"{name:>28} | {full:>+8.3f} | {cut:>+8.3f} | {faith:>7.2%} | "
                  f"{'OK' if g2 else 'LOW':>4} | {'OK' if feat_ok else 'BAD':>5}")
            sys.stdout.flush()
            del task, bundle, eng; gc.collect()
        except Exception as e:
            msg = str(e).encode("ascii", "replace").decode("ascii")[:60]
            results.append((name, False, False))
            print(f"{name:>28} | ERROR: {msg}"); sys.stdout.flush()

    n_g2 = sum(1 for _, g2, _ in results if g2)
    print(f"\nGate 2: {n_g2}/{len(results)} docstring tasks > 60%")


if __name__ == "__main__":
    main()
