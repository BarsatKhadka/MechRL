"""Test IOI with the PRECISE Wang et al. circuit, not the rough approximation.

Wang et al.'s circuit isn't just "all edges touching canonical heads" — it's a
specific directed pathway with channel-specific connections:
  - INPUT → previous_token heads (all channels Q, K, V)
  - INPUT → duplicate_token heads (Q, K, V)
  - INPUT → s2_inhibition heads (Q only)
  - INPUT → negative / name mover / backup name mover heads (K, V only)
  - previous_token → induction (K, V)
  - induction → s2_inhibition (K, V)
  - duplicate_token → s2_inhibition (K, V)
  - s2_inhibition → name movers / backup / negative (Q only)
  - name movers / backup / negative → OUTPUT

This is what the paper claims is ~87% faithful. Let's verify.
"""

from __future__ import annotations

import torch

from mechrl.tasks import IOITask
from mechrl.env import build_graph, AblationEngine


IOI_CIRCUIT = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2),
                          (9, 7), (9, 0), (11, 9)],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

# Wang et al.'s precise connections — (sender_group, receiver_group, channels)
PRECISE_CONNECTIONS = [
    ("INPUT", "previous token", ("q", "k", "v")),
    ("INPUT", "duplicate token", ("q", "k", "v")),
    ("INPUT", "s2 inhibition", ("q",)),
    ("INPUT", "negative", ("k", "v")),
    ("INPUT", "name mover", ("k", "v")),
    ("INPUT", "backup name mover", ("k", "v")),
    ("previous token", "induction", ("k", "v")),
    ("induction", "s2 inhibition", ("k", "v")),
    ("duplicate token", "s2 inhibition", ("k", "v")),
    ("s2 inhibition", "negative", ("q",)),
    ("s2 inhibition", "name mover", ("q",)),
    ("s2 inhibition", "backup name mover", ("q",)),
    ("negative", "OUTPUT", ()),
    ("name mover", "OUTPUT", ()),
    ("backup name mover", "OUTPUT", ()),
]


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


def group_heads(group_name):
    """Return the (layer, head) tuples for a circuit group."""
    if group_name == "INPUT":
        # Input acts as a "virtual" group — represented by input node + all MLPs
        # (MLPs read from residual stream same as input does in Wang's model)
        return None  # special handling
    if group_name == "OUTPUT":
        return None  # special handling
    return set(IOI_CIRCUIT[group_name])


def edge_in_precise_circuit(parent_parsed, child_parsed, child_qkv):
    """Check if edge (parent → child via channel) is in Wang et al.'s circuit."""
    for sender_group, receiver_group, channels in PRECISE_CONNECTIONS:
        # Sender side
        if sender_group == "INPUT":
            sender_ok = parent_parsed[0] in ("input", "mlp")
        elif sender_group == "OUTPUT":
            sender_ok = False  # OUTPUT can't be a sender
        else:
            sender_heads = group_heads(sender_group)
            sender_ok = (parent_parsed[0] == "head"
                        and (parent_parsed[1], parent_parsed[2]) in sender_heads)

        # Receiver side
        if receiver_group == "OUTPUT":
            receiver_ok = child_parsed[0] in ("logits", "mlp")
        elif receiver_group == "INPUT":
            receiver_ok = False
        else:
            receiver_heads = group_heads(receiver_group)
            receiver_ok = (child_parsed[0] == "head"
                          and (child_parsed[1], child_parsed[2]) in receiver_heads)

        # Channel side: only matters when child is an attention head
        if child_parsed[0] == "head":
            if not channels:  # OUTPUT receiver — no channel constraint
                channel_ok = True
            else:
                channel_ok = child_qkv in channels
        else:
            channel_ok = True  # MLPs / logits have no channel

        if sender_ok and receiver_ok and channel_ok:
            return True
    return False


def build_precise_mask(engine):
    mask = torch.zeros(engine.n_edges, dtype=torch.bool)
    for i, edge in enumerate(engine.edge_list):
        p = parse_node(edge.parent.name)
        c = parse_node(edge.child.name)
        if p is None or c is None:
            continue
        qkv = getattr(edge, "qkv", None)
        if edge_in_precise_circuit(p, c, qkv):
            mask[i] = True
    return mask


def build_random_mask(n_edges, n_alive_target, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_edges, generator=g)
    mask = torch.zeros(n_edges, dtype=torch.bool)
    mask[perm[:n_alive_target]] = True
    return mask


def main():
    print("Loading IOI task and engine...")
    task = IOITask(num_examples=30, device="cpu")
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph)

    full = engine.full_baseline()
    all_cut = engine.run_with_mask(engine.all_cut_mask())
    print(f"\nFull baseline:           {full:+.4f}")
    print(f"All cut (worst case):    {all_cut:+.4f}")
    print(f"Maximum possible drop:   {abs(full - all_cut):.4f}")

    print("\nBuilding precise Wang et al. circuit mask...")
    precise_mask = build_precise_mask(engine)
    n_alive = precise_mask.sum().item()
    print(f"  Edges in precise circuit: {n_alive}")
    print(f"  (Wang et al. report ~150-300 edges in their published circuit)")

    print("\nRunning precise circuit ablation...")
    score_precise = engine.run_with_mask(precise_mask)
    faith_precise = score_precise / full if full != 0 else float("nan")
    print(f"  score:        {score_precise:+.4f}")
    print(f"  faithfulness: {faith_precise:.2%}")
    print(f"  (Wang et al. report ~87%)")

    print(f"\nRandom subgraphs of same size ({n_alive} edges, 5 trials):")
    random_scores = []
    for trial in range(5):
        rmask = build_random_mask(engine.n_edges, n_alive, seed=trial)
        s = engine.run_with_mask(rmask)
        f = s / full if full != 0 else float("nan")
        random_scores.append(s)
        print(f"  trial {trial}: score {s:+.4f}  faithfulness {f:.2%}")
    mean_r = sum(random_scores) / len(random_scores)
    faith_r = mean_r / full if full != 0 else float("nan")
    print(f"  MEAN:    score {mean_r:+.4f}  faithfulness {faith_r:.2%}")

    print(f"\nVERDICT:")
    print(f"  Precise circuit faithfulness: {faith_precise:.2%}")
    print(f"  Random (same size):           {faith_r:.2%}")
    if faith_precise > 0.7:
        print(f"  [STRONG PASS] matches paper's claim of ~87%")
    elif faith_precise > 0.4:
        print(f"  [PARTIAL] meaningful signal but below paper's claim")
    else:
        print(f"  [WEAK] circuit doesn't preserve signal as expected")


if __name__ == "__main__":
    main()
