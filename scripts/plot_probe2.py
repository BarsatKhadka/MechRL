"""Probe-2 figure (§5.2): the agent's selection vs the EAP-IG ranking, and whether the
low-ranked edges it rescues are load-bearing. Reads runs/interaction_edges.json.

    python -m scripts.plot_probe2 --in runs/interaction_edges.json --out figures/fig_probe2
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = {"IOITask": "IOI", "GreaterThanOriginal": "Greater-than", "DocstringGPT2Task": "Docstring",
       "GenderedPronounTask": "Gendered pronoun", "SubjectVerbAgreementTask": "Subject-verb",
       "AcronymTask": "Acronyms", "SimpleSyllogismTask": "Simple syllogism",
       "OppositeSyllogismTask": "Opposite syllogism", "CountryCapitalTask": "Country-capital",
       "MCQAnchoredBiasTask": "Multiple choice"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="runs/interaction_edges.json")
    p.add_argument("--out", default="figures/fig_probe2")
    args = p.parse_args()

    d = sorted(json.load(open(args.inp))["rows"], key=lambda r: r["faith_gap_vs_topC"])
    labs = [LAB.get(r["task"], r["task"]) for r in d]
    y = np.arange(len(d))

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif", "axes.edgecolor": "#2b2b2b"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.7))

    # (a) agent vs top-|C| faith, dumbbell; agent coloured win / tie / loss
    for i, r in enumerate(d):
        a, t, g = r["faith_agent"], r["faith_topC"], r["faith_gap_vs_topC"]
        c = "#2e9e5b" if g > 0.01 else ("#c0395b" if g < -0.01 else "#888")
        axA.plot([t, a], [i, i], color="#c8c8c8", lw=2.2, zorder=1)
        axA.scatter(t, i, facecolors="white", edgecolors="#555", s=64, lw=1.7, zorder=3)
        axA.scatter(a, i, color=c, s=78, zorder=3)
    axA.scatter([], [], facecolors="white", edgecolors="#555", s=64, lw=1.7, label="top-$|C|$ by score")
    axA.scatter([], [], color="#2e9e5b", s=78, label="agent (more faithful)")
    axA.scatter([], [], color="#c0395b", s=78, label="agent (less faithful)")
    axA.set_yticks(y); axA.set_yticklabels(labs)
    axA.set_xlabel("faithfulness $f$ at matched size $|C|$")
    axA.set_title("(a)  agent's selection vs the attribution ranking", fontsize=11.5)
    axA.legend(loc="lower right", frameon=True, edgecolor="#cccccc", fontsize=9)
    axA.grid(axis="x", ls=":", color="#cfcfcf", lw=0.8); axA.set_axisbelow(True)
    axA.spines[["top", "right"]].set_visible(False)

    # (b) faith drop when the discordant (below-cutoff) edges are ablated
    drop = [r["faith_drop_from_discordant"] for r in d]
    cols = ["#2a6f9e" if v >= 0 else "#c0395b" for v in drop]
    axB.barh(y, drop, color=cols, zorder=3, height=0.62)
    axB.axvline(0, color="#2b2b2b", lw=0.9)
    axB.set_yticks(y); axB.set_yticklabels([])
    axB.set_xlabel("faithfulness drop when discordant edges ablated")
    axB.set_title("(b)  are the rescued below-cutoff edges load-bearing?", fontsize=11.5)
    axB.grid(axis="x", ls=":", color="#cfcfcf", lw=0.8); axB.set_axisbelow(True)
    axB.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"saved {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
