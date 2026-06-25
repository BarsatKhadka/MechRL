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
    p.add_argument("--tau", type=float, default=0.003, help="ACDC threshold to report")
    p.add_argument("--out", default="runs/baseline_summary.json")
    args = p.parse_args()

    S = {"acdc": {}, "eapig": {}, "acdc_tau": args.tau}

    for f in sorted(glob.glob(os.path.join(args.runs_dir, "acdc_*.json"))):
        n = os.path.basename(f)
        if "circuit" in n or "smoke" in n or "copysupp" in n.lower():   # skip per-tau circuit dumps / smoke / copy-supp
            continue
        task = n[len("acdc_"):-len(".json")]
        res = json.load(open(f)).get("results", [])
        if not res:
            continue
        r = min(res, key=lambda x: abs(x["tau"] - args.tau))            # closest threshold to --tau
        S["acdc"][task] = {"edges": r["final_edges"], "faith": r["final_faith"],
                           "passes": r["forward_passes"], "tau": r["tau"]}

    for f in sorted(glob.glob(os.path.join(args.runs_dir, "eapig_*.json"))):
        task = os.path.basename(f)[len("eapig_"):-len(".json")]
        S["eapig"][task] = {"curve": json.load(open(f))["curve"]}       # [(edges, faith), ...]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(S, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    print("acdc tasks :", sorted(S["acdc"]))
    print("eapig tasks:", sorted(S["eapig"]))


if __name__ == "__main__":
    main()
