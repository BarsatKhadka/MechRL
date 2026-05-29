"""Stage 2 validation: do canonical literature circuits score higher than random?

For each task with published canonical heads:
  1. Build a "canonical-touching" mask: keep edges where at least one endpoint
     is a canonical head (or input/logits/MLP that's part of the published
     circuit). Cut everything else.
  2. Build random masks of the same size.
  3. Run AblationEngine on both, compare to full baseline.

If canonical mask preserves the metric and random masks don't, then:
  (a) The ablation engine is correct (intervention actually does something).
  (b) The literature's canonical heads are doing what they're claimed to do.
  (c) Our setup is consistent with published mech interp.

This is the most important sanity check before any RL training.
"""

from __future__ import annotations

import torch

from mechrl.tasks import IOITask, GreaterThanTask, InductionTask
from mechrl.env import build_graph, AblationEngine


# Canonical heads per task (verifiable against papers)
IOI_CANONICAL = {
    (9, 9), (10, 0), (9, 6),
    (10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9),
    (10, 7), (11, 10),
    (7, 3), (7, 9), (8, 6), (8, 10),
    (5, 5), (5, 8), (5, 9), (6, 9),
    (0, 1), (0, 10), (3, 0),
    (2, 2), (4, 11),
}

GREATERTHAN_CANONICAL = {
    (0, 1), (0, 3), (0, 5),
    (5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1),
}
GREATERTHAN_CANONICAL_MLPS = {0, 1, 2, 3, 8, 9, 10, 11}

INDUCTION_CANONICAL = {
    (5, 1), (5, 5), (6, 9), (7, 2), (7, 10),
}


def parse_node(name: str):
    """Return ('head', layer, head) or ('mlp', layer, -) or ('input',) or ('logits',)."""
    if name == "input":
        return ("input",)
    if name == "logits":
        return ("logits",)
    if name.startswith("a"):
        try:
            l, h = name.split(".")
            return ("head", int(l[1:]), int(h[1:]))
        except (ValueError, IndexError):
            return None
    if name.startswith("m"):
        try:
            return ("mlp", int(name[1:]))
        except ValueError:
            return None
    return None


def build_canonical_mask(engine, canonical_heads, canonical_mlps=None):
    """Mask where True = edge keeps clean activation, False = swap with corrupted.

    True if BOTH endpoints are "in" the canonical circuit:
      - attention head in canonical_heads
      - MLP layer in canonical_mlps
      - input or logits node (boundary)
    """
    canonical_mlps = canonical_mlps or set()
    mask = torch.zeros(engine.n_edges, dtype=torch.bool)
    for i, edge in enumerate(engine.edge_list):
        p, c = parse_node(edge.parent.name), parse_node(edge.child.name)
        if p is None or c is None:
            continue
        if _in_canonical(p, canonical_heads, canonical_mlps) and \
           _in_canonical(c, canonical_heads, canonical_mlps):
            mask[i] = True
    return mask


def build_canonical_touching_mask(engine, canonical_heads, canonical_mlps=None):
    """Looser variant: keep edges where AT LEAST ONE endpoint is canonical.
    Includes all edges that flow info INTO or OUT OF canonical components.
    """
    canonical_mlps = canonical_mlps or set()
    mask = torch.zeros(engine.n_edges, dtype=torch.bool)
    for i, edge in enumerate(engine.edge_list):
        p, c = parse_node(edge.parent.name), parse_node(edge.child.name)
        if p is None or c is None:
            continue
        if _in_canonical(p, canonical_heads, canonical_mlps) or \
           _in_canonical(c, canonical_heads, canonical_mlps):
            mask[i] = True
    return mask


def _in_canonical(node_parsed, heads, mlps):
    if node_parsed[0] == "head":
        return (node_parsed[1], node_parsed[2]) in heads
    if node_parsed[0] == "mlp":
        return node_parsed[1] in mlps
    return node_parsed[0] in ("input", "logits")


def build_random_mask(engine, n_alive_target, seed=0):
    """Mask with exactly n_alive_target edges set to True, the rest False."""
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(engine.n_edges, generator=g)
    mask = torch.zeros(engine.n_edges, dtype=torch.bool)
    mask[perm[:n_alive_target]] = True
    return mask


def faithfulness(score, full_baseline):
    """Fraction of full baseline preserved. 1.0 = identical to full model."""
    if full_baseline == 0:
        return float("nan")
    return score / full_baseline


def test_task(task_name, task_cls, task_kwargs, canonical_heads,
              canonical_mlps=None, n_random_trials=5):
    print(f"\n{'='*70}\nTASK: {task_name}\n{'='*70}")

    task = task_cls(device="cpu", **task_kwargs)
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)

    full = engine.full_baseline()
    print(f"  Full baseline:           {full:+.4f}")

    all_cut = engine.run_with_mask(engine.all_cut_mask())
    print(f"  All cut (worst case):    {all_cut:+.4f}")
    print(f"  -> max possible drop: {abs(full - all_cut):.4f}")

    # CANONICAL: both endpoints must be canonical
    canon_strict = build_canonical_mask(engine, canonical_heads, canonical_mlps)
    n_canon_strict = canon_strict.sum().item()
    score_canon_strict = engine.run_with_mask(canon_strict)
    print(f"\n  CANONICAL (strict — both endpoints canonical):")
    print(f"    edges alive: {n_canon_strict:>5}")
    print(f"    score:       {score_canon_strict:+.4f}")
    print(f"    faithfulness: {faithfulness(score_canon_strict, full):.2%}")

    # CANONICAL TOUCHING: at least one endpoint is canonical
    canon_touch = build_canonical_touching_mask(engine, canonical_heads, canonical_mlps)
    n_canon_touch = canon_touch.sum().item()
    score_canon_touch = engine.run_with_mask(canon_touch)
    print(f"\n  CANONICAL (touching — at least one endpoint canonical):")
    print(f"    edges alive: {n_canon_touch:>5}")
    print(f"    score:       {score_canon_touch:+.4f}")
    print(f"    faithfulness: {faithfulness(score_canon_touch, full):.2%}")

    # RANDOM at same size as canonical-touching
    print(f"\n  RANDOM (same edge count as canonical-touching, {n_random_trials} trials):")
    random_scores = []
    for trial in range(n_random_trials):
        rmask = build_random_mask(engine, n_canon_touch, seed=trial)
        score = engine.run_with_mask(rmask)
        faith = faithfulness(score, full)
        random_scores.append(score)
        print(f"    trial {trial}: edges alive {rmask.sum().item():>5}  score {score:+.4f}  faithfulness {faith:.2%}")
    mean_random = sum(random_scores) / len(random_scores)
    print(f"    MEAN:    score {mean_random:+.4f}  faithfulness {faithfulness(mean_random, full):.2%}")

    # Verdict
    print(f"\n  VERDICT:")
    canon_faith = faithfulness(score_canon_touch, full)
    rand_faith = faithfulness(mean_random, full)
    print(f"    canonical-touching faithfulness: {canon_faith:.2%}")
    print(f"    random (same size) faithfulness: {rand_faith:.2%}")
    if canon_faith > rand_faith + 0.10:
        print(f"    [PASS] canonical is meaningfully better than random by {(canon_faith - rand_faith):.2%}")
    elif canon_faith > rand_faith:
        print(f"    [WEAK PASS] canonical is better than random by only {(canon_faith - rand_faith):.2%}")
    else:
        print(f"    [FAIL] canonical is NOT better than random")


def main():
    test_task("ioi", IOITask, {"num_examples": 30}, IOI_CANONICAL)
    test_task("greaterthan", GreaterThanTask, {"num_examples": 20},
              GREATERTHAN_CANONICAL, GREATERTHAN_CANONICAL_MLPS)
    test_task("induction", InductionTask, {"num_examples": 20, "half_len": 25},
              INDUCTION_CANONICAL)


if __name__ == "__main__":
    main()
