"""Test whether a hand-built Olsson-style induction circuit beats top-3000.

If our hand-built minimal circuit (induction heads + previous-token heads +
specific connections) gives meaningfully higher faithfulness than EAP-IG's
top-3000, then EAP-IG is missing the right edges and we need a better
prefilter for induction.

If hand-built ≈ top-3000, then induction is genuinely distributed.

Comparisons:
  - Full alive (100%)
  - All cut (0%)
  - EAP-IG top-3000
  - Induction heads only (5 heads, all touching edges)
  - Induction + previous-token heads + MLPs (Olsson-style circuit)
  - Random subgraph of same size as the targeted circuit
"""

from __future__ import annotations

import torch

from mechrl.tasks import InductionTask
from mechrl.env import build_graph, Prefilter, AblationEngine


# Olsson-style induction circuit components
INDUCTION_HEADS = {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}
# Previous-token head candidates (less established than induction)
# These are the early-layer heads commonly noted as previous-token markers
PREVIOUS_TOKEN_HEADS = {(0, 4), (1, 4), (2, 2), (3, 3), (4, 11)}
# All MLPs (Olsson notes MLPs help integrate residual stream)
ALL_MLPS = set(range(12))


def parse_node(name: str):
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


def in_circuit(node_parsed, heads, mlps):
    if node_parsed is None:
        return False
    if node_parsed[0] == "head":
        return (node_parsed[1], node_parsed[2]) in heads
    if node_parsed[0] == "mlp":
        return node_parsed[1] in mlps
    return node_parsed[0] in ("input", "logits")


def build_circuit_mask(engine, heads, mlps):
    """Keep edges where BOTH endpoints are in the circuit."""
    mask = torch.zeros(engine.n_edges, dtype=torch.bool)
    for i, edge in enumerate(engine.edge_list):
        p = parse_node(edge.parent.name)
        c = parse_node(edge.child.name)
        if in_circuit(p, heads, mlps) and in_circuit(c, heads, mlps):
            mask[i] = True
    return mask


def build_touching_mask(engine, heads, mlps):
    """Keep edges where AT LEAST ONE endpoint is in the circuit."""
    mask = torch.zeros(engine.n_edges, dtype=torch.bool)
    for i, edge in enumerate(engine.edge_list):
        p = parse_node(edge.parent.name)
        c = parse_node(edge.child.name)
        if in_circuit(p, heads, mlps) or in_circuit(c, heads, mlps):
            mask[i] = True
    return mask


def build_random_mask(engine, n_alive, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(engine.n_edges, generator=g)
    mask = torch.zeros(engine.n_edges, dtype=torch.bool)
    mask[perm[:n_alive]] = True
    return mask


def main():
    print("Loading induction task and computing prefilter...")
    task = InductionTask(num_examples=20, half_len=8, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)
    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)

    full = engine.full_baseline()
    cut = engine.corrupted_baseline()
    print(f"\nFull baseline:  {full:+.4f}  (faithfulness 100%)")
    print(f"Cut baseline:   {cut:+.4f}  (faithfulness 0%)")

    print(f"\n  {'mask':>45} | {'edges':>6} | {'faithfulness':>12}")
    print(f"  {'-'*45} | {'-'*6} | {'-'*12}")

    def report(name, mask):
        n = mask.sum().item()
        f = engine.faithfulness(mask)
        print(f"  {name:>45} | {n:>6} | {f:>11.2%}")

    # 1. Reference points
    report("all alive (full model)", engine.all_alive_mask())
    report("all cut (corrupted)", engine.all_cut_mask())

    # 2. EAP-IG top-3000
    report("EAP-IG top-3000", pref.candidate_mask(3000))

    # 3. Just induction heads (strict + touching)
    only_ind = build_circuit_mask(engine, INDUCTION_HEADS, set())
    report("induction heads only (strict, both endpoints)", only_ind)
    only_ind_touch = build_touching_mask(engine, INDUCTION_HEADS, set())
    report("induction heads only (touching)", only_ind_touch)

    # 4. Induction + previous-token heads + MLPs (Olsson-style full circuit)
    olsson = INDUCTION_HEADS | PREVIOUS_TOKEN_HEADS
    olsson_strict = build_circuit_mask(engine, olsson, ALL_MLPS)
    report("Olsson circuit + MLPs (strict)", olsson_strict)
    olsson_touch = build_touching_mask(engine, olsson, ALL_MLPS)
    report("Olsson circuit + MLPs (touching)", olsson_touch)

    # 5. Random subgraphs of matching sizes
    n_olsson_touch = olsson_touch.sum().item()
    print(f"\n  Random subgraphs of size {n_olsson_touch} (matching Olsson-touching):")
    random_scores = []
    for trial in range(3):
        rmask = build_random_mask(engine, n_olsson_touch, seed=trial)
        f = engine.faithfulness(rmask)
        random_scores.append(f)
        print(f"    trial {trial}: faithfulness {f:>6.2%}")
    print(f"    MEAN: {sum(random_scores)/len(random_scores):>6.2%}")

    print(f"\n  Random subgraphs of size 3000 (matching top-3000):")
    for trial in range(3):
        rmask = build_random_mask(engine, 3000, seed=trial)
        f = engine.faithfulness(rmask)
        print(f"    trial {trial}: faithfulness {f:>6.2%}")


if __name__ == "__main__":
    main()
