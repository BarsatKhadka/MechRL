"""KL-faithfulness sweep over all 13 training tasks.

For each task: build the KL engine and report top-3k KL-faith (how well the
candidate set reproduces the full model's OUTPUT DISTRIBUTION) + the KL all-cut
baseline. num_examples is pinned to each task's cached prefilter so nothing
recomputes (avoids the memory spike that crashed earlier sweeps).

Sensibility check: is top-3k KL-faith reasonably high (candidate set reproduces
the model) for ALL tasks, or does some task need a bigger K under KL?
"""

import gc
import sys

from mechrl.tasks import (
    IOITask, IOIAfterOpener, IOINoPlaceObject, IOIFriendsFound,
    GreaterThanOriginal, GreaterThanReversed, GreaterThanBeganEnded, GreaterThanTookPlace,
    DocstringGPT2Task, DocstringGPT2Sphinx7Task, DocstringGPT2Google5Task,
    DocstringGPT2ClassSphinxTask, DocstringGPT2Numpy5Task,
)
from mechrl.env import TaskBundle

# (class, num_examples) — pinned to the cached prefilters
TASKS = [
    (IOITask, 20),
    (IOIAfterOpener, 30), (IOINoPlaceObject, 30), (IOIFriendsFound, 30),
    (GreaterThanOriginal, 30), (GreaterThanReversed, 30),
    (GreaterThanBeganEnded, 30), (GreaterThanTookPlace, 30),
    (DocstringGPT2Task, 20), (DocstringGPT2Sphinx7Task, 20), (DocstringGPT2Google5Task, 20),
    (DocstringGPT2ClassSphinxTask, 20), (DocstringGPT2Numpy5Task, 20),
]


def main():
    hdr = f"{'task':>28} | {'KL all-cut':>10} | {'KL-faith top-3k':>15}"
    print(hdr); print("-" * len(hdr)); sys.stdout.flush()
    results = []
    for cls, n in TASKS:
        name = cls.__name__
        try:
            task = cls(num_examples=n, device="cpu")
            bundle = TaskBundle.build(task, k=3000)
            eng = bundle.engine
            kl_cut = eng.corrupted_baseline()
            kl_faith = eng.faithfulness(bundle.candidate_mask)
            results.append((name, kl_faith))
            print(f"{name:>28} | {kl_cut:>10.3f} | {kl_faith:>14.4f}")
            sys.stdout.flush()
            del task, bundle, eng
            gc.collect()
        except Exception as e:
            msg = str(e).encode("ascii", "replace").decode("ascii")[:60]
            results.append((name, None))
            print(f"{name:>28} | ERROR: {msg}")
            sys.stdout.flush()

    print("\n=== summary ===")
    ok = [(n, f) for n, f in results if f is not None]
    if ok:
        lo = min(ok, key=lambda x: x[1])
        hi = max(ok, key=lambda x: x[1])
        avg = sum(f for _, f in ok) / len(ok)
        print(f"KL-faith top-3k: min={lo[1]:.3f} ({lo[0]}), max={hi[1]:.3f} ({hi[0]}), mean={avg:.3f}")
        low = [n for n, f in ok if f < 0.7]
        print(f"tasks with KL-faith < 0.70 (may need bigger K): {low if low else 'none'}")


if __name__ == "__main__":
    main()
