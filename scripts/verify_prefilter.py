"""Verify the EAP-IG prefilter is doing what we claim.

This script makes the prefilter output FULLY TRANSPARENT:
  1. Loads canonical IOI heads directly from ACDC's source code (which
     vendored Wang et al. 2022 Table 2 — verifiable against the paper).
  2. Runs EAP-IG attribution on real IOI prompts.
  3. Lists EVERY canonical head with its rank in the attribution scores.
  4. Lists the top-30 highest-scoring heads regardless of canonical status.
  5. Writes a CSV of all head scores so you can cross-check externally.

Important distinction:
  - GPT-2 small ALWAYS has 144 attention heads (12 layers × 12 heads).
  - The prefilter doesn't keep heads or throw away heads.
  - It scores ~32k EDGES and keeps the top-K edges.
  - A head "appears in top-K" if at least one of its edges is in top-K.
  - At K=3000, ~138 of 144 heads appear in candidates — agent's action space
    can touch edges connected to almost any head.

Run:
    venv\\Scripts\\python.exe scripts\\verify_prefilter.py
"""

from __future__ import annotations

import csv
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from eap.attribute import attribute
from eap.graph import Graph

# Import directly from ACDC's source — this is Wang et al.'s Table 2 as vendored
# by Conmy et al. 2023. Verifiable against IOI paper arxiv 2211.00593.
from acdc.ioi.utils import IOI_CIRCUIT

from mechrl.tasks import IOITask


# ---- Data loader (eap-ig format) ----

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    return list(clean), list(corrupted), torch.stack(labels)


class IOIEapDataset(Dataset):
    def __init__(self, ioi_task: IOITask):
        batch = ioi_task.validation_batch()
        model = ioi_task.model
        self.clean_strs = [model.to_string(batch.clean_tokens[i]) for i in range(batch.batch_size)]
        self.corrupt_strs = [model.to_string(batch.corrupted_tokens[i]) for i in range(batch.batch_size)]

        io_ids = batch.correct_labels.tolist()
        tokenizer = model.tokenizer
        s_ids = []
        for i, prompt in enumerate(self.clean_strs):
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            seen, s_id = set(), None
            for tok in tokens:
                if tok in seen:
                    s_id = tok
                    break
                seen.add(tok)
            s_ids.append(s_id if s_id is not None else io_ids[i])

        self.labels = torch.tensor(list(zip(io_ids, s_ids)), dtype=torch.long)

    def __len__(self):
        return len(self.clean_strs)

    def __getitem__(self, i):
        return self.clean_strs[i], self.corrupt_strs[i], self.labels[i]


def logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    last_logits = logits[idx, input_length - 1]
    labels = labels.to(last_logits.device)
    good_bad = torch.gather(last_logits, -1, labels)
    diff = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        diff = -diff
    if mean:
        diff = diff.mean()
    return diff


# ---- Score extraction utilities ----

def parse_head(name: str):
    """Parse node name like 'a9.h6' → (9, 6). Return None if not an attention head."""
    if not name.startswith("a"):
        return None
    try:
        layer_part, head_part = name.split(".")
        return (int(layer_part[1:]), int(head_part[1:]))
    except (ValueError, IndexError):
        return None


def edge_max_scores_by_head(graph: Graph):
    """Return dict {(layer, head): max abs(score) across all edges touching that head}."""
    by_head: dict = {}
    for edge in graph.edges.values():
        score = edge.score.item() if torch.is_tensor(edge.score) else float(edge.score)
        abs_score = abs(score)
        for node in (edge.parent, edge.child):
            h = parse_head(node.name)
            if h is not None:
                if h not in by_head or abs_score > by_head[h]:
                    by_head[h] = abs_score
    return by_head


def ranked_heads_by_top_edge_score(graph: Graph):
    """Return list of ((layer, head), max_edge_score) sorted by score descending."""
    by_head = edge_max_scores_by_head(graph)
    return sorted(by_head.items(), key=lambda x: -x[1])


# ---- Main verification ----

def main():
    print("=" * 70)
    print("STEP 1: Load canonical IOI heads from ACDC's IOI_CIRCUIT")
    print("=" * 70)
    print("Source: paperCodes/Automatic-Circuit-Discovery/acdc/ioi/utils.py")
    print("This is Wang et al. 2022 Table 2 as vendored by Conmy et al.")
    print("You can verify against arxiv 2211.00593 Table 2.\n")

    canonical_heads_by_category = {}
    canonical_heads_all = set()
    for category, heads in IOI_CIRCUIT.items():
        canonical_heads_by_category[category] = list(heads)
        canonical_heads_all.update(heads)

    for category, heads in canonical_heads_by_category.items():
        print(f"  {category}: {heads}")
    print(f"\n  Total unique canonical heads: {len(canonical_heads_all)}")

    print("\n" + "=" * 70)
    print("STEP 2: Load model and IOI dataset")
    print("=" * 70)
    ioi = IOITask(num_examples=50, device="cpu")
    model = ioi.model
    print(f"  model: {model.cfg.model_name}")
    print(f"  total attention heads in GPT-2 small: "
          f"{model.cfg.n_layers * model.cfg.n_heads} "
          f"(= {model.cfg.n_layers} layers × {model.cfg.n_heads} heads)")
    print(f"  (these 144 heads ALWAYS exist — prefilter doesn't add/remove them)")

    print("\n" + "=" * 70)
    print("STEP 3: Build EAP-IG graph and run attribution")
    print("=" * 70)
    graph = Graph.from_model(model)
    print(f"  graph edges: {len(graph.edges)}")
    print(f"  graph nodes: {len(graph.nodes)}")
    print(f"  (nodes = 144 attn heads + 12 MLPs + 1 input + 1 logits = 158)")

    ds = IOIEapDataset(ioi)
    loader = DataLoader(ds, batch_size=10, collate_fn=collate_EAP)
    print(f"\n  Running EAP-IG attribution on {len(ds)} prompts (5 IG steps)...")
    attribute(model, graph, loader,
              partial(logit_diff, loss=True, mean=True),
              method="EAP-IG-inputs", ig_steps=5)
    print("  Done.")

    print("\n" + "=" * 70)
    print("STEP 4: Rank EVERY head by its highest-scoring edge")
    print("=" * 70)
    ranked = ranked_heads_by_top_edge_score(graph)
    print(f"  Total heads with at least one scored edge: {len(ranked)}")
    print(f"  (Heads not appearing means none of their edges were given any score —")
    print(f"   should be 0 for GPT-2 small under EAP-IG)\n")

    print("  Top 30 heads by max edge score:")
    print(f"  {'rank':>4} | {'head':>10} | {'max edge score':>15} | {'canonical?':>10}")
    print(f"  {'-'*4} | {'-'*10} | {'-'*15} | {'-'*10}")
    for rank, (head, score) in enumerate(ranked[:30], start=1):
        is_canon = "YES" if head in canonical_heads_all else "-"
        print(f"  {rank:>4} | L{head[0]:>2}.H{head[1]:<2}     | {score:>15.6f} | {is_canon:>10}")

    print("\n" + "=" * 70)
    print("STEP 5: Rank of EACH canonical head")
    print("=" * 70)
    head_to_rank = {h: i + 1 for i, (h, _) in enumerate(ranked)}
    print(f"  {'category':>20} | {'head':>10} | {'rank (of 144)':>13}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*13}")
    for category, heads in canonical_heads_by_category.items():
        for h in heads:
            rank = head_to_rank.get(h, "NOT FOUND")
            print(f"  {category:>20} | L{h[0]:>2}.H{h[1]:<2}     | {rank!s:>13}")

    print("\n" + "=" * 70)
    print("STEP 6: Top-K edge retention test")
    print("=" * 70)
    print("  For each K, list canonical heads that appear in at least one")
    print("  edge of the top-K edges.\n")

    def heads_in_top_k_edges(k):
        edges = sorted(graph.edges.values(),
                       key=lambda e: -abs(e.score.item() if torch.is_tensor(e.score) else float(e.score)))[:k]
        heads = set()
        for e in edges:
            for n in (e.parent, e.child):
                h = parse_head(n.name)
                if h is not None:
                    heads.add(h)
        return heads

    print(f"  {'K':>6} | {'heads in top-K':>14} | {'canonical retained':>20}")
    print(f"  {'-'*6} | {'-'*14} | {'-'*20}")
    for k in [100, 300, 500, 1000, 2000, 3000, 5000]:
        heads = heads_in_top_k_edges(k)
        retained = canonical_heads_all & heads
        missing = canonical_heads_all - retained
        line = f"  {k:>6} | {len(heads):>14} | {len(retained):>3}/{len(canonical_heads_all):<3}"
        if missing:
            line += f"  missing: {sorted(missing)}"
        print(line)

    print("\n" + "=" * 70)
    print("STEP 7: Dump all head scores to CSV for external verification")
    print("=" * 70)
    csv_path = Path(__file__).resolve().parents[1] / "datasets_dump" / "ioi_head_scores.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "layer", "head", "max_edge_score", "is_canonical", "category"])
        head_to_category = {}
        for category, heads in canonical_heads_by_category.items():
            for h in heads:
                head_to_category[h] = category
        for rank, (head, score) in enumerate(ranked, start=1):
            is_canon = head in canonical_heads_all
            cat = head_to_category.get(head, "")
            w.writerow([rank, head[0], head[1], f"{score:.8f}", is_canon, cat])
    print(f"  wrote {csv_path.relative_to(Path(__file__).resolve().parents[1])}")
    print(f"  -> open this file to see EVERY head's score and rank.")


if __name__ == "__main__":
    main()
