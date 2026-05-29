"""Check if induction's low top-3000 faithfulness is fixed by larger K."""

from mechrl.tasks import InductionTask
from mechrl.env import build_graph, Prefilter, AblationEngine


def main():
    task = InductionTask(num_examples=20, half_len=25, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)

    print(f"\nInduction faithfulness vs K:")
    print(f"  {'K':>6} | {'faithfulness':>12}")
    print(f"  {'-'*6} | {'-'*12}")
    for k in [500, 1000, 3000, 5000, 10000, 20000, 32491]:
        if k > engine.n_edges:
            k = engine.n_edges
        mask = pref.candidate_mask(k)
        f = engine.faithfulness(mask)
        print(f"  {k:>6} | {f:>11.2%}")


if __name__ == "__main__":
    main()
