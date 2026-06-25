"""EAP-IG greedy circuit search (Hanna et al. 2024, Appendix E) on OUR graph + KL engine.

Why port instead of wrapping paperCodes/eap-ig: like run_acdc_greedy, this keeps the baseline on
the EXACT same harness as our agent -- same EAP-IG scores (the prefilter already computes them),
same corrupted-patching ablation, same KL faithfulness -- so size/faith/cost are directly
comparable in Table 4. The only thing that differs from our agent is the search: greedy over the
attribution scores (EAP-IG) versus the learned policy.

Greedy (Appendix E, faithful to Graph.apply_greedy): start with only the logits in the circuit;
repeatedly add the highest-|score| edge whose CHILD is already in the circuit, and bring its parent
(and the parent's incoming edges) into the frontier. The selection itself costs no forward passes
(it reads precomputed scores); we spend one forward pass per circuit size we evaluate, plus the
one-time attribution cost -- a few passes total, far below ACDC's ~10^4 per behaviour.

We sweep a set of sizes n, greedily select n edges, and measure (edges, faith) for each, writing
the curve + per-size circuits. Read off "EAP-IG greedy's size at faith ~0.9" for the table.

Usage (GPU; one-time EAP-IG attribution then cheap evals):
    python -m scripts.run_eapig_greedy --task IOITask --device cuda \
        --sizes 100 200 400 700 1000 1500 2000 --num-examples 20 --out runs/eapig_ioi.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from mechrl.tasks import (
    IOITask, GreaterThanOriginal, DocstringGPT2Task,
    CopySuppressionTask, GenderedPronounTask, SubjectVerbAgreementTask,
    AcronymTask, SimpleSyllogismTask, OppositeSyllogismTask,
    MCQAnchoredBiasTask, CountryCapitalTask,
)
from mechrl.env import AblationEngine, Prefilter, build_graph

TASKS = {"IOITask": IOITask, "GreaterThanOriginal": GreaterThanOriginal,
         "DocstringGPT2Task": DocstringGPT2Task,
         "CopySuppressionTask": CopySuppressionTask,
         "GenderedPronounTask": GenderedPronounTask,
         "SubjectVerbAgreementTask": SubjectVerbAgreementTask,
         "AcronymTask": AcronymTask, "SimpleSyllogismTask": SimpleSyllogismTask,
         "OppositeSyllogismTask": OppositeSyllogismTask,
         "MCQAnchoredBiasTask": MCQAnchoredBiasTask, "CountryCapitalTask": CountryCapitalTask}


def greedy_select(engine, n_edges):
    """Appendix-E greedy: pick the n highest-|score| edges reachable backward from the logits.
    Returns a bool mask over engine.edge_list. No forward passes -- pure score bookkeeping."""
    edges = engine.edge_list
    absc = [abs(float(e.score)) for e in edges]
    incoming = defaultdict(list)                       # child node name -> [edge idx]
    for i, e in enumerate(edges):
        incoming[e.child.name].append(i)

    mask = torch.zeros(len(edges), dtype=torch.bool)
    in_nodes = {"logits"}
    heap, pushed = [], set()
    def push_incoming(node):
        for j in incoming.get(node, []):
            if j not in pushed:
                pushed.add(j)
                heapq.heappush(heap, (-absc[j], j))
    push_incoming("logits")

    added = 0
    while added < n_edges and heap:
        _, i = heapq.heappop(heap)
        if mask[i]:
            continue
        mask[i] = True
        added += 1
        pnode = edges[i].parent.name
        if pnode not in in_nodes:
            in_nodes.add(pnode)
            push_incoming(pnode)
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="IOITask", choices=list(TASKS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument("--ig-steps", type=int, default=5)
    p.add_argument("--prefilter-metric", choices=["task", "kl"], default=None,
                   help="attribution target: None=logit-diff (donor default), 'kl' for MCQ.")
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[100, 200, 400, 700, 1000, 1500, 2000])
    p.add_argument("--out", default="runs/eapig_greedy.json")
    args = p.parse_args()

    print(f"Loading {args.task} (n={args.num_examples}) on {args.device}...", flush=True)
    task = TASKS[args.task](num_examples=args.num_examples, device=args.device)
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph, metric_type="kl")

    pm = "kl" if args.prefilter_metric == "kl" else None
    print(f"Scoring edges with EAP-IG ({pm or 'logit-diff'}, ig_steps={args.ig_steps})...", flush=True)
    pref = Prefilter(task, graph, ig_steps=args.ig_steps, metric_type=pm)
    pref.compute()                                     # writes EAP-IG scores into graph edges

    t0 = time.time()
    curve, circuits = [], {}
    passes = 0
    for n in sorted(args.sizes):
        mask = greedy_select(engine, n)                # no forward passes
        faith = float(engine.faithfulness(mask)); passes += 1
        kept = int(mask.sum().item())
        curve.append((kept, faith))
        circuits[str(kept)] = [engine.edge_list[i].name for i in mask.nonzero(as_tuple=True)[0].tolist()]
        print(f"  n={n:>5} -> {kept:>5} edges  faith {faith:.4f}", flush=True)

    payload = {
        "task": args.task, "num_examples": args.num_examples, "ig_steps": args.ig_steps,
        "prefilter_metric": pm or "logitdiff",
        "n_edges_total": len(engine.edge_list),
        "faith_eval_passes": passes,                   # greedy selection is forward-pass-free
        "elapsed_sec": time.time() - t0,
        "curve": curve,                                # [(edges, faith), ...]
        "circuits": circuits,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}  (faith evals: {passes}; attribution: ~{args.ig_steps} fwd+bwd, one-time)", flush=True)


if __name__ == "__main__":
    main()
