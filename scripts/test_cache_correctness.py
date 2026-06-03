"""Verify the corrupted-pass cache is CORRECT (identical faithfulness) and FASTER.

Correctness is non-negotiable: if the cached path gave even slightly different
faithfulness, every training reward would be subtly wrong. So we compare, on the
SAME masks, an engine WITH cache vs an engine WITHOUT cache and require equality.
"""

import time
import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import build_graph, Prefilter, AblationEngine


def main():
    print("Building IOI task + graph...")
    task = IOITask(num_examples=20, device="cpu")
    graph = build_graph(task.model)
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)

    # a handful of distinct masks to test
    masks = [
        pref.candidate_mask(3000),
        pref.candidate_mask(1000),
        pref.candidate_mask(500),
    ]
    # plus a couple of partial cuts
    m = pref.candidate_mask(3000).clone()
    idx = m.nonzero(as_tuple=True)[0]
    m[idx[:200]] = False
    masks.append(m)

    eng_nocache = AblationEngine(task, graph)
    eng_nocache.use_corrupted_cache = False

    eng_cache = AblationEngine(task, graph)
    eng_cache.use_corrupted_cache = True

    print("\n--- correctness: cached vs uncached faithfulness ---")
    max_diff = 0.0
    for i, mk in enumerate(masks):
        f_no = eng_nocache.faithfulness(mk)
        f_ca = eng_cache.faithfulness(mk)
        d = abs(f_no - f_ca)
        max_diff = max(max_diff, d)
        print(f"  mask {i}: nocache={f_no:.6f}  cache={f_ca:.6f}  diff={d:.2e}")
    print(f"max diff = {max_diff:.2e}")
    assert max_diff < 1e-4, "CACHE CHANGES THE RESULT — do not use it!"
    print("PASS: cache is numerically identical.")

    # determinism within the cached engine (reuse path stable)
    a = eng_cache.faithfulness(masks[0])
    b = eng_cache.faithfulness(masks[0])
    assert abs(a - b) < 1e-6, "cached engine not deterministic across calls"

    # --- speed: many sequential evals (simulating an episode) ---
    print("\n--- speed: 20 sequential faithfulness calls ---")
    test_mask = pref.candidate_mask(3000)

    eng_nocache._corrupted_baseline = None  # force fresh
    t0 = time.time()
    for _ in range(20):
        eng_nocache.faithfulness(test_mask)
    t_no = time.time() - t0

    # warm the cache once, then time
    eng_cache.faithfulness(test_mask)
    t0 = time.time()
    for _ in range(20):
        eng_cache.faithfulness(test_mask)
    t_ca = time.time() - t0

    print(f"  no cache: {t_no:.2f}s  ({t_no/20*1000:.0f} ms/call)")
    print(f"  cache   : {t_ca:.2f}s  ({t_ca/20*1000:.0f} ms/call)")
    print(f"  speedup : {t_no/t_ca:.2f}x")


if __name__ == "__main__":
    main()
