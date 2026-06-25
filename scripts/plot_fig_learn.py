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
    p.add_argument("--probe-warm", dest="probe_warm", default="runs/interaction_edges_warm.json")
    p.add_argument("--out", default="figures/fig_learn")
    p.add_argument("--kstar", type=int, default=3000)
    args = p.parse_args()

    pf = json.load(open(args.prefilter)); ks = pf["ks"]; T = pf["tasks"]
    pb = sorted(json.load(open(args.probe))["rows"], key=lambda r: r["faith_gap_vs_topC"])

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.edgecolor": "#2b2b2b"})
    fig = plt.figure(figsize=(13.6, 7.7))
    gs = GridSpec(2, 2, width_ratios=[1, 1.18], height_ratios=[1, 1], hspace=0.34, wspace=0.14)
    axK = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[1, 0])
    axA = fig.add_subplot(gs[0:2, 1], projection="polar")

    # (a) candidate-set faithfulness vs K  (colours keyed by the legend, panel d)
    for n in ORDER:
        if n not in T:
            continue
        cur = T[n].get("kl-attr" if n in KL else "logitdiff", {})
        axK.plot(ks, [cur.get(str(k)) for k in ks], "-o", color=COLORS.get(n, "#888"),
                 lw=1.5, ms=3.5, alpha=0.95, label=LAB.get(n, n))
    axK.axhline(0.9, color="#1a1a1a", ls="--", lw=1.1)
    axK.text(max(ks) - 650, 0.862, "$f = 0.9$", color="#1a1a1a", ha="center", va="center",
             fontsize=9.5, fontweight="medium")
    axK.axvline(args.kstar, color="#2b2b2b", ls=":", lw=1.5)
    axK.text(args.kstar - 120, 0.45, f"chosen $K={args.kstar}$", color="#2b2b2b", ha="right",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    axK.set_xlabel("$K$ (candidate edges)"); axK.set_ylabel("candidate-set faithfulness")
    axK.set_ylim(0, 1.04); axK.set_xlim(0, max(ks) + 250)
    axK.grid(color="#cfcfcf", lw=0.8, ls=":"); axK.set_axisbelow(True)
    axK.spines[["top", "right"]].set_visible(False)
    axK.set_title("(a)  candidate-set faithfulness vs $K$", fontsize=11.5)
    legK = axK.legend(title="behaviour", loc="lower right", frameon=True, edgecolor="#000000",
                      facecolor="white", framealpha=1.0, fontsize=7.3, title_fontsize=8.3,
                      ncol=2, labelspacing=0.3, columnspacing=0.9, handletextpad=0.4, borderpad=0.6)
    legK.get_frame().set_linewidth(1.0)

    # (b) spider web: spoke = behaviour, radius = agent_f - top-|C|_f (gain over the ranking).
    # The green ring outside the dashed tie-circle is the "agent more faithful" zone; the web
    # bulges out where the agent beats the score and dents inside the ring only at subject-verb.
    present = [t for t in ORDER if any(r["task"] == t for r in pb)]
    by_task = {r["task"]: r for r in pb}
    gap = np.array([by_task[t]["faith_gap_vs_topC"] for t in present])
    ang = np.linspace(0, 2 * np.pi, len(present), endpoint=False)
    ac = np.concatenate([ang, ang[:1]]); gc = np.concatenate([gap, gap[:1]])
    th = np.linspace(0, 2 * np.pi, 200); rmax = 0.135
    axA.set_theta_zero_location("N"); axA.set_theta_direction(-1)
    axA.set_rorigin(-0.085); axA.set_ylim(-0.075, rmax)
    axA.fill_between(th, 0, rmax, color="#2e9e5b", alpha=0.11, zorder=0)        # more-faithful zone
    axA.fill_between(th, -0.075, 0, color="#c0395b", alpha=0.06, zorder=0)      # less-faithful zone
    axA.plot(th, np.zeros_like(th), color="#2b2b2b", lw=1.1, ls="--", zorder=2)  # tie ring (agent = ranking)
    axA.fill(ac, gc, color="#2e9e5b", alpha=0.13, zorder=2)
    axA.plot(ac, gc, color="#5b6b78", lw=1.6, zorder=3)
    for a, t in zip(ang, present):
        g = by_task[t]["faith_gap_vs_topC"]
        axA.scatter(a, g, color=COLORS[t], s=160, edgecolors="white", lw=1.2, zorder=5)
        lab = f"{g:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
        axA.text(a, g + 0.0125, lab, fontsize=7.3, ha="center", va="center", color="#222", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.7))
    # optional: warm-start gain as a hollow dot joined to the zero-shot dot by a dotted spoke
    import os
    if os.path.exists(args.probe_warm):
        warm = {r["task"]: r["faith_gap_vs_topC"] for r in json.load(open(args.probe_warm))["rows"]}
        for a, t in zip(ang, present):
            if t in warm:
                axA.plot([a, a], [by_task[t]["faith_gap_vs_topC"], warm[t]], color=COLORS[t],
                         ls=":", lw=1.5, zorder=4)
                axA.scatter(a, warm[t], facecolors="white", edgecolors=COLORS[t], s=120, lw=1.9, zorder=6)
        from matplotlib.lines import Line2D
        axA.legend(handles=[Line2D([0], [0], marker="o", color="#555", linestyle="none", label="zero-shot"),
                            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#555",
                                   linestyle="none", label="warm-start")],
                   loc="upper left", bbox_to_anchor=(-0.05, 1.12), frameon=False, fontsize=8.5)
    _botang = ang[0] + np.pi + (ang[1] - ang[0]) / 2   # bottom between-spoke angle (both labels here)
    axA.text(_botang, 0.108, "agent\nmore faithful", ha="center", va="center", fontsize=8.5,
             color="#2e7d32", style="italic", zorder=6)
    axA.text(_botang, -0.048, "ranking\nmore faithful", ha="center", va="center", fontsize=8,
             color="#b3445c", style="italic", zorder=6)
    axA.set_xticks(ang); axA.set_xticklabels([])
    axA.set_yticks([0.0, 0.05, 0.10]); axA.set_yticklabels(["0", ".05", ".10"], fontsize=7, color="#777")
    axA.grid(color="#dcdcdc", lw=0.7)
    axA.set_title("(b)  the agent's gain in faithfulness over the attribution ranking", fontsize=12, pad=20)

    # (c) simple bars: faithfulness lost when each behaviour's rescued (discordant) edges are
    # deleted -- positive = load-bearing; the simple syllogism is the one negative (harmful).
    oc = sorted(pb, key=lambda r: r["faith_drop_from_discordant"])
    for i, r in enumerate(oc):
        axB.barh(i, r["faith_drop_from_discordant"], color=COLORS[r["task"]], height=0.66, zorder=3)
    axB.axvline(0, color="#2b2b2b", lw=0.9)
    axB.set_yticks(range(len(oc))); axB.set_yticklabels([LAB[r["task"]] for r in oc], fontsize=8)
    axB.set_ylim(-0.7, len(oc) - 0.3)
    axB.set_xlabel("faithfulness lost when the rescued edges are deleted")
    axB.set_title("(c)  the rescued edges are load-bearing", fontsize=11.5)
    axB.grid(axis="x", ls=":", color="#cfcfcf", lw=0.8); axB.set_axisbelow(True)
    axB.spines[["top", "right"]].set_visible(False)


    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"saved {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
