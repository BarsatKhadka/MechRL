"""Combined §5.2 figure (full TMLR page width): (a) candidate-set faithfulness vs K on the
left, and on the right the two probe panels -- (b) agent vs top-|C| selection and (c) whether
the rescued below-cutoff edges are load-bearing. Reads runs/prefilter_ceiling.json and
runs/interaction_edges.json.

    python -m scripts.plot_fig_learn --out figures/fig_learn
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

LAB = {"IOITask": "IOI", "GreaterThanOriginal": "Greater-than", "DocstringGPT2Task": "Docstring",
       "GenderedPronounTask": "Gendered pronoun", "SubjectVerbAgreementTask": "Subject-verb",
       "AcronymTask": "Acronyms", "SimpleSyllogismTask": "Simple syllogism",
       "OppositeSyllogismTask": "Opposite syllogism", "CountryCapitalTask": "Country-capital",
       "MCQAnchoredBiasTask": "Multiple choice"}
COLORS = {"GenderedPronounTask": "#3b4cc0", "AcronymTask": "#ef8636", "SimpleSyllogismTask": "#8e44ad",
          "OppositeSyllogismTask": "#d6453d", "SubjectVerbAgreementTask": "#2e9e5b",
          "MCQAnchoredBiasTask": "#1f8a8a", "CountryCapitalTask": "#c9a227",
          "IOITask": "#444444", "GreaterThanOriginal": "#e377c2", "DocstringGPT2Task": "#8c564b"}
KL = {"MCQAnchoredBiasTask"}
ORDER = list(LAB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefilter", default="runs/prefilter_ceiling.json")
    p.add_argument("--probe", default="runs/interaction_edges.json")
    p.add_argument("--out", default="figures/fig_learn")
    p.add_argument("--kstar", type=int, default=3000)
    args = p.parse_args()

    pf = json.load(open(args.prefilter)); ks = pf["ks"]; T = pf["tasks"]
    pb = sorted(json.load(open(args.probe))["rows"], key=lambda r: r["faith_gap_vs_topC"])

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.edgecolor": "#2b2b2b"})
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.34, wspace=0.16)
    axK = fig.add_subplot(gs[0, 0]); axA = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 0]); axL = fig.add_subplot(gs[1, 1]); axL.axis("off")

    # (a) candidate-set faithfulness vs K  (colours keyed by the legend, panel d)
    for n in ORDER:
        if n not in T:
            continue
        cur = T[n].get("kl-attr" if n in KL else "logitdiff", {})
        axK.plot(ks, [cur.get(str(k)) for k in ks], "-o", color=COLORS.get(n, "#888"),
                 lw=1.5, ms=3.5, alpha=0.95)
    axK.axhline(0.9, color="#1a1a1a", ls="--", lw=1.1)
    axK.text(max(ks) - 650, 0.862, "$f = 0.9$", color="#1a1a1a", ha="center", va="center",
             fontsize=9.5, fontweight="medium")
    axK.axvline(args.kstar, color="#2b2b2b", ls=":", lw=1.5)
    axK.text(args.kstar, 0.05, f"chosen $K={args.kstar}$", color="#2b2b2b", ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    axK.set_xlabel("$K$ (candidate edges)"); axK.set_ylabel("candidate-set faithfulness")
    axK.set_ylim(0, 1.04); axK.set_xlim(0, max(ks) + 250)
    axK.grid(color="#cfcfcf", lw=0.8, ls=":"); axK.set_axisbelow(True)
    axK.spines[["top", "right"]].set_visible(False)
    axK.set_title("(a)  candidate-set faithfulness vs $K$", fontsize=11.5)

    y = np.arange(len(pb))
    # (b) agent vs top-|C|: dumbbell coloured per behaviour, win/loss read from direction
    for i, r in enumerate(pb):
        col = COLORS[r["task"]]
        axA.plot([r["faith_topC"], r["faith_agent"]], [i, i], color=col, lw=2.0, alpha=0.45, zorder=1)
        axA.scatter(r["faith_topC"], i, facecolors="white", edgecolors=col, s=46, lw=1.7, zorder=3)
        axA.scatter(r["faith_agent"], i, color=col, s=58, zorder=3)
    axA.scatter([], [], facecolors="white", edgecolors="#555", s=46, lw=1.7, label="top-$|C|$ by score")
    axA.scatter([], [], color="#555", s=58, label="agent")
    axA.set_yticks([]); axA.set_ylim(-0.7, len(pb) - 0.3)
    axA.set_xlabel("$f$ at matched size $|C|$")
    axA.set_title("(b)  agent's selection vs top-$|C|$ by score", fontsize=11.5)
    axA.legend(loc="lower right", frameon=False, fontsize=8.5)
    axA.grid(axis="x", ls=":", color="#cfcfcf", lw=0.8); axA.set_axisbelow(True)
    axA.spines[["top", "right"]].set_visible(False)

    # (c) load-bearing: bars coloured per behaviour, same row order as (b); sign = direction
    for i, r in enumerate(pb):
        axB.barh(i, r["faith_drop_from_discordant"], color=COLORS[r["task"]], height=0.62, zorder=3)
    axB.axvline(0, color="#2b2b2b", lw=0.9)
    axB.set_yticks([]); axB.set_ylim(-0.7, len(pb) - 0.3)
    axB.set_xlabel("faith drop when discordant edges ablated")
    axB.set_title("(c)  are the rescued below-cutoff edges load-bearing?", fontsize=11.5)
    axB.grid(axis="x", ls=":", color="#cfcfcf", lw=0.8); axB.set_axisbelow(True)
    axB.spines[["top", "right"]].set_visible(False)

    # shared behaviour colour key (fills the empty bottom-right)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[n],
                      markeredgecolor=COLORS[n], markersize=9, label=LAB[n]) for n in ORDER]
    lg = axL.legend(handles=handles, title="behaviour", loc="center", frameon=True,
                    edgecolor="#000000", facecolor="white", framealpha=1.0, fontsize=10.5,
                    title_fontsize=11.5, ncol=2, labelspacing=0.7, columnspacing=1.4,
                    handletextpad=0.5, borderpad=1.0)
    lg.get_frame().set_linewidth(1.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"saved {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
