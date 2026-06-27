"""Section 5.3 figure.

(a) The policy's OWN compute budget (within MechRL, no baselines): one rollout is ~150 steps =
    ~150 passes; a deployment is best-of-16 (17 rollouts) ~= 2.5k passes per behaviour; a held-out
    behaviour additionally pays a one-time warm-start adaptation (~80 iters); and the policy itself
    is trained once (1200 iters x 512 steps ~= 6e5 passes) and then reused for every behaviour.
(b) The cost--minimality plane: x = forward passes to produce the circuit (log), y = circuit size
    |C| at f>=0.9, one marker per behaviour, shaped by method. EAP-IG (free selection) forms a
    cheap but size-scattered left column, the policy a fixed-cost middle column, ACDC the small-but-
    expensive right spread. Bottom-left is ideal; open markers never reached f=0.9.

    python -m scripts.plot_fig_cost --out figures/fig_cost
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables projection="3d")

LAB = {"IOITask": "IOI", "GreaterThanOriginal": "Greater-than", "DocstringGPT2Task": "Docstring",
       "GenderedPronounTask": "Gendered", "SubjectVerbAgreementTask": "Subject-verb",
       "AcronymTask": "Acronyms", "SimpleSyllogismTask": "Simple syll.",
       "OppositeSyllogismTask": "Opposite syll.", "CountryCapitalTask": "Country",
       "MCQAnchoredBiasTask": "MCQ"}
ORDER = list(LAB)
POLICY = {"IOITask": (1123, 0.96), "GreaterThanOriginal": (464, 0.98), "DocstringGPT2Task": (725, 0.89),
          "GenderedPronounTask": (390, 0.99), "SubjectVerbAgreementTask": (440, 0.94),
          "AcronymTask": (445, 0.96), "SimpleSyllogismTask": (290, 0.92),
          "OppositeSyllogismTask": (700, 0.94), "CountryCapitalTask": (650, 0.89),
          "MCQAnchoredBiasTask": (700, 0.93)}
C_ACDC, C_POL, C_EAP = "#3d4451", "#2e6fb0", "#e0992e"
PANEL_BG = "#eef0f4"
EAP_PASSES = 14  # ~5 attribution + 1 eval per size; selection is forward-pass-free


def acdc_pt(rows, thr=0.9):
    ok = [r for r in rows if r["faith"] >= thr]
    return (min(ok, key=lambda r: r["edges"]), True) if ok else (max(rows, key=lambda r: r["faith"]), False)


def eap_best(S, task, thr=0.9):
    """EAP-IG's best attribution target: (size, faith, reached) -- smallest circuit reaching
    f>=0.9 over ld and kl; else the plateau point at K with its faithfulness."""
    cand = [(e, f) for key in ("eapig", "eapig_kl") for e, f in S[key][task]["curve"] if f >= thr]
    if cand:
        e, f = min(cand, key=lambda p: p[0])
        return e, f, True
    last = max((S["eapig"][task]["curve"][-1], S["eapig_kl"][task]["curve"][-1]), key=lambda p: p[1])
    return last[0], last[1], False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="runs/baseline_summary.json")
    p.add_argument("--passes-a", default="runs/zeroshot_passes_a.json")
    p.add_argument("--passes-b", default="runs/zeroshot_passes_b.json")
    p.add_argument("--out", default="figures/fig_cost")
    args = p.parse_args()

    S = json.load(open(args.summary))
    pol_pass = {}
    for fn in (args.passes_a, args.passes_b):
        for r in json.load(open(fn))["rows"]:
            pol_pass[r["task"]] = r["forward_passes_total"]
    deploy = int(np.mean(list(pol_pass.values())))            # ~2475

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.edgecolor": "#2b2b2b",
                         "text.color": "#2b2b2b", "axes.labelcolor": "#3a3a3a"})
    fig = plt.figure(figsize=(12.8, 4.9))
    gs = GridSpec(1, 2, width_ratios=[1, 1.18], wspace=0.2)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

    # ---- (a) the policy's own compute budget: a multiplicative ladder ----
    levels = [("1 step", 1, "#d3e3f3"),
              ("1 rollout", 152, "#9ec4e6"),
              ("deploy / behaviour", deploy, C_POL),
              ("warm-start adapt", 80 * 512, "#caa05a"),
              ("policy training", 1200 * 512, "#555555")]
    y = np.arange(len(levels))[::-1]                          # step at top, training at bottom
    axA.barh(y, [l[1] for l in levels], height=0.6, color=[l[2] for l in levels], zorder=3)
    for yi, (_, v, _) in zip(y, levels):
        lab = (f"{v/1000:.1f}k" if v < 10000 else f"{v/1000:.0f}k") if v >= 1000 else f"{v}"
        axA.text(v * 1.5, yi, lab, va="center", ha="left", fontsize=9, fontweight="medium")
    # the multiplicative jumps, drawn between the bars
    for yi, txt in ((y[1] + 0.5, "$\\times$150 steps"),
                    (y[2] + 0.5, "$\\times$17 rollouts\n(best-of-16)")):
        axA.text(1.4, yi, txt, va="center", ha="left", fontsize=7.6, color="#33506b", style="italic")
    axA.axhline(1.5, color="#bbbbbb", lw=0.9, ls="--", zorder=1)   # per-behaviour | one-time divider
    axA.text(2.6e6, 3.1, "paid per\nbehaviour", fontsize=7.6, color="#33506b",
             ha="right", va="center", style="italic")
    axA.text(2.6e6, 0.4, "paid\nonce", fontsize=7.6, color="#5a5a5a",
             ha="right", va="center", style="italic")
    axA.set_xscale("log"); axA.set_xlim(0.6, 3.0e6)
    axA.set_yticks(y); axA.set_yticklabels([l[0] for l in levels], fontsize=9)
    axA.set_ylim(-0.7, len(levels) - 0.3)
    axA.set_xlabel("forward passes (log)")
    axA.set_title("(a)  the policy's compute budget", fontsize=11.5, fontweight="medium", pad=10)
    axA.set_facecolor(PANEL_BG)
    axA.grid(axis="x", color="white", lw=1.4); axA.set_axisbelow(True)
    for s in axA.spines.values():
        s.set_visible(False)
    axA.tick_params(colors="#6a6a6a", length=0)

    # ---- (b) cost vs size, gray-panel styled; best = bottom-left ----
    POL_RED = "#e23b41"
    XMIN, XMAX, YMAX = 8, 30000, 3150
    valid = []
    for t in ORDER:
        a, aok = acdc_pt(S["acdc"][t])
        axB.scatter(a["passes"], a["edges"], marker="s", s=82,
                    facecolors=C_ACDC if aok else PANEL_BG, edgecolors="white" if aok else C_ACDC,
                    linewidths=1.4, zorder=5)
        if aok: valid.append((a["passes"], a["edges"]))
        e, ef, eok = eap_best(S, t)
        axB.scatter(EAP_PASSES, e, marker="^", s=104,
                    facecolors=C_EAP if eok else PANEL_BG, edgecolors="white" if eok else C_EAP,
                    linewidths=1.4, zorder=5)
        if eok: valid.append((EAP_PASSES, e))
        axB.scatter(pol_pass[t], POLICY[t][0], marker="*", s=320,
                    facecolors=POL_RED, edgecolors="white", linewidths=1.1, zorder=6)
        valid.append((pol_pass[t], POLICY[t][0]))

    axB.set_xscale("log"); axB.set_xlim(XMIN, XMAX); axB.set_ylim(0, YMAX)
    axB.set_yticks([0, 1000, 2000, 3000])
    axB.set_xlabel("forward passes to produce circuit  (log) $\\;\\rightarrow$ costlier")
    axB.set_ylabel("circuit size $|C|\\;\\rightarrow$ larger")
    axB.set_title("(b)  cost vs. size", fontsize=11.5, fontweight="medium", pad=10)
    axB.set_facecolor(PANEL_BG)
    axB.grid(color="white", lw=1.2); axB.set_axisbelow(True)
    for s in axB.spines.values():
        s.set_visible(False)
    axB.tick_params(colors="#6a6a6a", length=0)
    axB.legend(handles=[Line2D([0], [0], marker="*", color="none", markerfacecolor=POL_RED,
                               markeredgecolor="white", markersize=16, label="Policy (ours)"),
                        Line2D([0], [0], marker="^", color="none", markerfacecolor=C_EAP,
                               markeredgecolor="white", markersize=10, label="EAP-IG greedy"),
                        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_ACDC,
                               markeredgecolor="white", markersize=9, label="ACDC"),
                        Line2D([0], [0], marker="o", color="#9a9a9a", markerfacecolor=PANEL_BG,
                               markeredgecolor="#9a9a9a", linestyle="none", markersize=9,
                               label="open: never $f\\geq0.9$")],
               loc="upper right", fontsize=8.2, frameon=True, facecolor="white",
               edgecolor="none", framealpha=0.9, labelspacing=0.4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"saved {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
