"""Collapse the big ACDC + EAP-IG-greedy result JSONs into one small summary for Table 4 (§5.3).

The ACDC JSONs are multi-MB (they store every edge); we only need size/faith/passes per task at
a chosen threshold, plus EAP-IG-greedy's (edges, faith) curve. This writes a tiny JSON that can be
git-couriered to the laptop to build the table. Pure json -- no torch, safe on a login node.

    python -m scripts.baseline_summary            # -> runs/baseline_summary.json
    python -m scripts.baseline_summary --tau 0.003 --out runs/baseline_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="runs/baseline_summary.json")
    args = p.parse_args()

    S = {"acdc": {}, "eapig": {}, "eapig_kl": {}}

    # ACDC: keep ALL thresholds per task (list), so we can pick a non-degenerate point downstream.
    for f in sorted(glob.glob(os.path.join(args.runs_dir, "acdc_*.json"))):
        n = os.path.basename(f)
        if "circuit" in n or "smoke" in n or "copysupp" in n.lower():
            continue
        task = n[len("acdc_"):-len(".json")]
        res = json.load(open(f)).get("results", [])
        if res:
            S["acdc"][task] = [{"tau": r["tau"], "edges": r["final_edges"],
                                "faith": r["final_faith"], "passes": r["forward_passes"]} for r in res]

    # EAP-IG greedy curves: logit-diff (eapig_*) and KL (eapig_kl_*) attribution, kept separate.
    for f in sorted(glob.glob(os.path.join(args.runs_dir, "eapig_kl_*.json"))):
        task = os.path.basename(f)[len("eapig_kl_"):-len(".json")]
        S["eapig_kl"][task] = {"curve": json.load(open(f))["curve"]}
    for f in sorted(glob.glob(os.path.join(args.runs_dir, "eapig_*.json"))):
        n = os.path.basename(f)
        if n.startswith("eapig_kl_"):
            continue
        task = n[len("eapig_"):-len(".json")]
        S["eapig"][task] = {"curve": json.load(open(f))["curve"]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(S, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    print("acdc tasks    :", sorted(S["acdc"]))
    print("eapig tasks   :", sorted(S["eapig"]))
    print("eapig_kl tasks:", sorted(S["eapig_kl"]))


if __name__ == "__main__":
    main()
