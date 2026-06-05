"""ACDC (Conmy et al. 2023, Algorithm 1) run on OUR graph + OUR KL engine.

Why reimplement instead of running Conmy's code: this puts ACDC on the EXACT
same evaluation harness as our RL agent -- same EAP-IG graph, same corrupted-
patching ablation, same KL-divergence faithfulness. The ONLY difference left is
the search strategy (greedy threshold-pruning vs learned policy), which is what
we want to isolate. It also avoids the fragile cross-framework edge mapping that
broke the canonical-circuit scoring.

Algorithm 1 (faithful):
  H <- full graph; sort receivers output->input (reverse topological)
  for each receiver v (output->input):
      for each incoming edge (w -> v), parents later-layer first:
          tentatively cut it; one forward pass; measure KL rise
          if KL(circuit) rose by < tau:  remove permanently
          else:                          restore (edge matters)
  return H

Each edge test = one forward pass. Total forward passes ~= number of edges
tested = the headline cost (~32k for GPT-2). We log the (edges, faith) curve as
edges are removed, and report the final circuit per threshold. To find "ACDC's
size at faith 0.9", run a few thresholds and read off the point nearest 0.9.

Reachability pruning (matches ACDC's redundant-node removal): when we reach a
receiver that no longer has any live outgoing edge, it is disconnected from the
output, so its incoming edges are dropped for free (no forward pass) -- this is
why the realized pass count is somewhat below the 32k upper bound.

Usage (GPU strongly recommended -- ~32k forward passes per threshold):
    python -m scripts.run_acdc_greedy --device cuda \
        --thresholds 0.0005 0.001 0.003 0.01 0.03 \
        --num-examples 20 --out runs/acdc_ioi.json

Local smoke test (cheap -- only processes the first few receivers):
    python -m scripts.run_acdc_greedy --device cpu --thresholds 0.01 \
        --smoke-receivers 2 --out runs/acdc_smoke.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from mechrl.tasks import IOITask
from mechrl.env import build_graph, AblationEngine


TASKS = {"IOITask": IOITask}


def parse_dst(d: str):
    """'a5.h3<q>' -> ('a5.h3', 'q');  'm5' -> ('m5', None);  'logits' -> ('logits', None)."""
    if "<" in d:
        base, ch = d[:-1].split("<")
        return base, ch
    return d, None


def run_acdc(engine, graph, tau, smoke_receivers=0, log_every=500, verbose=True):
    """One ACDC run at threshold tau. Returns dict with final circuit + curve."""
    edges = engine.edge_list                       # mask index i <-> edges[i]
    n_edges = len(edges)

    # Group incoming edges per receiver (child, qkv); group outgoing per sender.
    incoming = defaultdict(list)                    # (child_name, qkv) -> [idx]
    outgoing = defaultdict(list)                    # parent_name -> [idx]
    for i, e in enumerate(edges):
        incoming[(e.child.name, e.qkv)].append(i)
        outgoing[e.parent.name].append(i)

    # Order parents within a receiver: later-layer (higher forward index) first,
    # matching ACDC's lexicographic-then-reversed parent order.
    fwd_idx = {i: graph.forward_index(edges[i].parent, attn_slice=False) for i in range(n_edges)}
    for key in incoming:
        incoming[key].sort(key=lambda i: fwd_idx[i], reverse=True)

    # Receivers in reverse-topological (output -> input) order.
    dst_order = [parse_dst(d) for d in reversed(graph.get_dst_nodes())]
    if smoke_receivers > 0:
        dst_order = dst_order[:smoke_receivers]

    mask = engine.all_alive_mask()
    full = engine.full_baseline()                  # KL of full model ~ 0
    cut = engine.corrupted_baseline()              # KL all-cut (max)
    denom = full - cut
    cur_kl = engine.run_with_mask(mask)            # ~0 ; full circuit
    passes = 1                                     # this forward pass counts

    def faith_of(kl):
        return (kl - cut) / denom if abs(denom) > 1e-9 else 0.0

    curve = [(int(mask.sum().item()), faith_of(cur_kl), passes)]
    t0 = time.time()

    for (child_name, qkv) in dst_order:
        # Reachability: a non-output receiver with no live outgoing edge is
        # disconnected -> drop its incoming edges for free (no forward pass).
        if child_name != "logits":
            if not any(bool(mask[j]) for j in outgoing.get(child_name, [])):
                for i in incoming[(child_name, qkv)]:
                    mask[i] = False
                continue

        for i in incoming[(child_name, qkv)]:
            if not bool(mask[i]):
                continue
            mask[i] = False                        # tentatively cut
            new_kl = engine.run_with_mask(mask)
            passes += 1
            if (new_kl - cur_kl) < tau:            # edge unimportant -> remove
                cur_kl = new_kl
                curve.append((int(mask.sum().item()), faith_of(cur_kl), passes))
            else:                                  # edge matters -> restore
                mask[i] = True
            if verbose and passes % log_every == 0:
                el = int(mask.sum().item())
                print(f"    [tau={tau}] passes={passes} edges={el} "
                      f"faith={faith_of(cur_kl):.4f} ({time.time()-t0:.0f}s)", flush=True)

    final_edges = int(mask.sum().item())
    return {
        "tau": tau,
        "final_edges": final_edges,
        "final_faith": faith_of(cur_kl),
        "final_kl": cur_kl,
        "forward_passes": passes,
        "kl_cut": cut,
        "kl_full": full,
        "n_edges_total": n_edges,
        "elapsed_sec": time.time() - t0,
        "curve": curve,                            # [(edges, faith, passes), ...]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="IOITask", choices=list(TASKS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.001, 0.003, 0.01])
    p.add_argument("--smoke-receivers", type=int, default=0,
                   help="if >0, only process the first N receivers (cheap local test)")
    p.add_argument("--out", default="runs/acdc_greedy.json")
    args = p.parse_args()

    print(f"Loading {args.task} (n={args.num_examples}) on {args.device}...", flush=True)
    task = TASKS[args.task](num_examples=args.num_examples, device=args.device)
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph, metric_type="kl")
    print(f"Graph: {engine.n_edges} edges. KL all-cut={engine.corrupted_baseline():.4f}", flush=True)

    results = []
    for tau in args.thresholds:
        print(f"\n=== ACDC greedy, tau={tau} ===", flush=True)
        r = run_acdc(engine, graph, tau, smoke_receivers=args.smoke_receivers)
        print(f"  -> {r['final_edges']} edges @ faith {r['final_faith']:.4f} "
              f"in {r['forward_passes']} forward passes ({r['elapsed_sec']:.0f}s)", flush=True)
        results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": args.task,
        "num_examples": args.num_examples,
        "device": args.device,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)

    print("\n=== SUMMARY (size vs faith) ===", flush=True)
    print(f"{'tau':>10} | {'edges':>7} | {'faith':>7} | {'fwd passes':>10}", flush=True)
    for r in results:
        print(f"{r['tau']:>10} | {r['final_edges']:>7} | {r['final_faith']:>7.4f} | {r['forward_passes']:>10}", flush=True)


if __name__ == "__main__":
    main()
