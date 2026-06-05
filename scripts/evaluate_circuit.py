"""Validity battery for a circuit (Gate 1).

Takes any circuit (a list of edge names, e.g. from extract_agent_circuit.py or a
saved ACDC run) and scores it on the same IOI engine, reporting:

  LEAD  ground-truth head recovery   -- which of Wang's 26 canonical heads appear
                                         in the circuit (the anti-circularity check:
                                         the reference was derived by humans, not
                                         from our KL reward).
  STRICT task accuracy               -- argmax agreement (does the circuit predict
                                         the SAME token as the full model, per
                                         example) + logit-diff recovery. Catches the
                                         per-example failures that mean-KL hides.
        circuit divergence           -- KL + normalized faithfulness (context).
        necessity (knockout)         -- cut ONLY the circuit from the full model;
                                         faith should drop (circuit carries signal).
        specificity                  -- random same-size circuit, for comparison.
  opt   ACDC overlap                 -- precision/recall/F1 of edges vs ACDC's
                                         circuit (pass --acdc <json>).

Usage:
    python -m scripts.evaluate_circuit --circuit runs/agent_circuit_ioi.json
    python -m scripts.evaluate_circuit --circuit runs/agent_circuit_ioi.json \
        --acdc runs/acdc_circuit_tau0.003.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechrl.tasks import IOITask
from mechrl.env import build_graph, AblationEngine

# Wang et al.'s 26 canonical IOI heads, by family.
IOI_FAMILIES = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}
IOI_CANONICAL = {h for hs in IOI_FAMILIES.values() for h in hs}


def parse_head(name: str):
    if name.startswith("a") and "." in name:
        try:
            l, h = name.split(".")
            return (int(l[1:]), int(h[1:]))
        except (ValueError, IndexError):
            return None
    return None


def mask_from_names(engine, names):
    name_to_idx = {e.name: i for i, e in enumerate(engine.edge_list)}
    mask = torch.zeros(engine.n_edges, dtype=torch.bool)
    missing = 0
    for nm in names:
        i = name_to_idx.get(nm)
        if i is None:
            missing += 1
        else:
            mask[i] = True
    return mask, missing


def circuit_logits(engine, mask):
    """Run the masked model and capture its raw logits (the engine's run_with_mask
    only returns the scalar KL, so we briefly swap in a capturing metric)."""
    cap = {}
    orig = engine.metric
    def cap_metric(logits, clean_logits, input_length, labels):
        cap["logits"] = logits.detach()
        cap["input_length"] = input_length
        return orig(logits, clean_logits, input_length, labels)
    engine.metric = cap_metric
    try:
        engine.run_with_mask(mask)
    finally:
        engine.metric = orig
    return cap["logits"], cap["input_length"]


def task_accuracy(engine, mask):
    """argmax agreement + logit-diff recovery vs the full model, per example."""
    full = engine._full_logits                       # [bs, seq, vocab]
    circ, inlen = circuit_logits(engine, mask)
    bs = circ.size(0)
    idx = torch.arange(bs, device=circ.device)
    pos = (inlen - 1).to(circ.device).long()
    full_last = full[:bs].to(circ.device)[idx, pos]  # [bs, vocab]
    circ_last = circ[idx, pos]

    argmax_match = (full_last.argmax(-1) == circ_last.argmax(-1)).float().mean().item()

    out = {"argmax_accuracy": argmax_match}
    labels = getattr(engine.dataloader.dataset, "labels", None)
    if labels is not None:
        labels = labels.to(circ.device)
        full_gb = full_last.gather(-1, labels)        # [bs, 2] = (correct, wrong)
        circ_gb = circ_last.gather(-1, labels)
        full_diff = (full_gb[:, 0] - full_gb[:, 1]).mean()
        circ_diff = (circ_gb[:, 0] - circ_gb[:, 1])
        out["logit_diff_recovery"] = (circ_diff.mean() / full_diff).item()
        out["correct_gt_wrong_frac"] = (circ_diff > 0).float().mean().item()
    return out


def random_mask(n_edges, n_alive, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_edges, generator=g)
    m = torch.zeros(n_edges, dtype=torch.bool)
    m[perm[:n_alive]] = True
    return m


def evaluate_circuit(engine, mask, acdc_names=None):
    n = int(mask.sum().item())
    kl = engine.run_with_mask(mask)
    kl_cut = engine.corrupted_baseline()
    faith = engine.faithfulness(mask)

    # ground-truth head recovery (LEAD)
    circuit_heads = set()
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        e = engine.edge_list[i]
        for nm in (e.parent.name, e.child.name):
            ph = parse_head(nm)
            if ph is not None:
                circuit_heads.add(ph)
    recovered = IOI_CANONICAL & circuit_heads
    families = {fam: sum((h in circuit_heads) for h in hs) for fam, hs in IOI_FAMILIES.items()}

    # task accuracy (STRICT)
    acc = task_accuracy(engine, mask)

    # necessity: knock the circuit out of the FULL model
    knock = engine.all_alive_mask()
    knock[mask] = False
    knockout_faith = engine.faithfulness(knock)

    # specificity: random same-size
    rmask = random_mask(engine.n_edges, n, seed=0)
    rand_faith = engine.faithfulness(rmask)
    rand_acc = task_accuracy(engine, rmask)["argmax_accuracy"]

    result = {
        "n_edges": n,
        "kl": kl, "kl_cut": kl_cut, "faith": faith,
        "heads_recovered": len(recovered),
        "heads_total": len(IOI_CANONICAL),
        "heads_by_family": families,
        "heads_recovered_list": sorted(recovered),
        **acc,
        "knockout_faith": knockout_faith,
        "random_same_size_faith": rand_faith,
        "random_same_size_argmax_acc": rand_acc,
    }

    if acdc_names is not None:
        acdc_mask, _ = mask_from_names(engine, acdc_names)
        inter = int((mask & acdc_mask).sum().item())
        a = int(mask.sum().item()); b = int(acdc_mask.sum().item())
        prec = inter / a if a else 0.0
        rec = inter / b if b else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        result["acdc_overlap"] = {"intersection": inter, "agent_edges": a,
                                  "acdc_edges": b, "precision": prec, "recall": rec, "f1": f1}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", required=True, help="circuit JSON with an 'edges' list of names")
    p.add_argument("--acdc", default=None, help="optional ACDC circuit JSON (edges list) for overlap")
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    circ = json.loads(Path(args.circuit).read_text())
    acdc_names = json.loads(Path(args.acdc).read_text())["edges"] if args.acdc else None

    print(f"Loading IOI engine (n={args.num_examples}) on {args.device}...", flush=True)
    task = IOITask(num_examples=args.num_examples, device=args.device)
    graph = build_graph(task.model)
    engine = AblationEngine(task, graph, metric_type="kl")

    mask, missing = mask_from_names(engine, circ["edges"])
    if missing:
        print(f"[warn] {missing} edge names not found in current graph (skipped)", flush=True)

    res = evaluate_circuit(engine, mask, acdc_names=acdc_names)

    print("\n=== VALIDITY BATTERY ===", flush=True)
    print(f"circuit: {res['n_edges']} edges", flush=True)
    print(f"\n[LEAD] ground-truth head recovery: {res['heads_recovered']}/{res['heads_total']} canonical heads", flush=True)
    for fam, hs in IOI_FAMILIES.items():
        print(f"    {fam:20s} {res['heads_by_family'][fam]}/{len(hs)}", flush=True)
    print(f"\n[STRICT] task accuracy:", flush=True)
    print(f"    argmax agreement vs full model: {res['argmax_accuracy']:.3f}", flush=True)
    if "logit_diff_recovery" in res:
        print(f"    logit-diff recovery:            {res['logit_diff_recovery']:.3f}", flush=True)
        print(f"    examples correct>wrong:         {res['correct_gt_wrong_frac']:.3f}", flush=True)
    print(f"\n[divergence] KL {res['kl']:.4f} (cut {res['kl_cut']:.4f})  faith {res['faith']:.4f}", flush=True)
    print(f"[necessity]  knockout faith (cut circuit from full): {res['knockout_faith']:.4f}  (lower = more necessary)", flush=True)
    print(f"[specificity] random same-size: faith {res['random_same_size_faith']:.4f}  argmax {res['random_same_size_argmax_acc']:.3f}", flush=True)
    if "acdc_overlap" in res:
        o = res["acdc_overlap"]
        print(f"\n[ACDC overlap] inter={o['intersection']}  precision={o['precision']:.3f}  recall={o['recall']:.3f}  F1={o['f1']:.3f}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
