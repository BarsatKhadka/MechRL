"""Compare patching vs mean ablation on IOI top-K masks.

Goal: see if mean ablation gives higher faithfulness (matching Wang et al.'s
~87% claim) than patching with corrupted prompts.
"""

from __future__ import annotations

import torch

from mechrl.tasks import IOITask
from mechrl.env import build_graph, Prefilter, AblationEngine


def main():
    task = IOITask(num_examples=30, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)

    full = engine.full_baseline()
    print(f"\nFull baseline: {full:+.4f}\n")

    print(f"  {'mask':>20} | {'patching':>10} | {'mean':>10} | {'mean-pos':>10}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")

    masks = [
        ("all alive", engine.all_alive_mask()),
        ("all cut", engine.all_cut_mask()),
        ("top-500", pref.candidate_mask(500)),
        ("top-1000", pref.candidate_mask(1000)),
        ("top-2000", pref.candidate_mask(2000)),
        ("top-3000", pref.candidate_mask(3000)),
        ("top-5000", pref.candidate_mask(5000)),
    ]

    for name, mask in masks:
        scores = {}
        for intervention in ["patching", "mean", "mean-positional"]:
            try:
                s = engine.run_with_mask(mask, intervention=intervention)
                scores[intervention] = s / full if full != 0 else float("nan")
            except Exception as e:
                scores[intervention] = None
        p_str = f"{scores['patching']:.2%}" if scores['patching'] is not None else "ERR"
        m_str = f"{scores['mean']:.2%}" if scores['mean'] is not None else "ERR"
        mp_str = f"{scores['mean-positional']:.2%}" if scores['mean-positional'] is not None else "ERR"
        print(f"  {name:>20} | {p_str:>10} | {m_str:>10} | {mp_str:>10}")


if __name__ == "__main__":
    main()
