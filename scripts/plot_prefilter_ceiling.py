"""Candidate-set faithfulness vs K per behaviour -- justifies the choice K=3000.

Reads the JSON from `verify_prefilter_kl --out`. For each behaviour it plots the KL
faithfulness of the top-K candidate set as K grows, and marks the chosen K=3000. The
prefilter selects edges by logit-diff attribution for every behaviour except
MCQAnchoredBiasTask (KL), so we plot the metric each behaviour actually uses. The curve
rises then plateaus; K=3000 sits at the knee -- near the ceiling, but far smaller than
the full ~32k-edge graph and a tractable action space for the agent.

    python -m scripts.plot_prefilter_ceiling --in runs/prefilter_ceiling.json \
        --out figures/fig_prefilter_k
"""
from __future__ import annotations
import argparse, json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KL_TASKS = {"MCQAnchoredBiasTask"}          # selects by KL attribution; others by logit-diff
LABELS = {
    "IOITask": "IOI", "GreaterThanOriginal": "Greater-than", "DocstringGPT2Task": "Docstring",
    "GenderedPronounTask": "Gendered pronoun", "SubjectVerbAgreementTask": "Subject-verb",
    "AcronymTask": "Acronyms", "SimpleSyllogismTask": "Simple syllogism",
    "OppositeSyllogismTask": "Opposite syllogism", "CountryCapitalTask": "Country-capital",
    "MCQAnchoredBiasTask": "Multiple choice",
}

# Per-behaviour colours: the seven held-out match fig_transfer; three training tasks added.
COLORS = {
    "GenderedPronounTask": "#3b4cc0", "AcronymTask": "#ef8636", "SimpleSyllogismTask": "#8e44ad",
    "OppositeSyllogismTask": "#d6453d", "SubjectVerbAgreementTask": "#2e9e5b",
    "MCQAnchoredBiasTask": "#1f8a8a", "CountryCapitalTask": "#c9a227",
    "IOITask": "#444444", "GreaterThanOriginal": "#e377c2", "DocstringGPT2Task": "#8c564b",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="runs/prefilter_ceiling.json")
    p.add_argument("--out", default="figures/fig_prefilter_k")
    p.add_argument("--kstar", type=int, default=3000)
    args = p.parse_args()

    d = json.load(open(args.inp))
    ks = d["ks"]
    tasks = d["tasks"]
    names = [n for n in LABELS if n in tasks] + [n for n in tasks if n not in LABELS]

    n = len(names); ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow),
                             sharex=True, sharey=True)
    for ax in axes.flat:
        ax.axis("off")
    for i, name in enumerate(names):
        ax = axes.flat[i]; ax.axis("on")
        attr = "kl-attr" if name in KL_TASKS else "logitdiff"
        cur = tasks[name].get(attr, {})
        ys = [cur.get(str(k)) for k in ks]
        xs = [k for k, y in zip(ks, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, "-o", color="#258", lw=2, ms=4)
        ax.axvline(args.kstar, color="#c44", lw=1.2, ls=":")
        ystar = cur.get(str(args.kstar))
        if ystar is not None:
            ax.text(args.kstar, 0.06, f"$K={args.kstar}$\n$f={ystar:.2f}$",
                    color="#c44", fontsize=7.5, ha="center", va="bottom")
        ax.set_title(LABELS.get(name, name), fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.15)
        if i % ncol == 0:
            ax.set_ylabel("candidate-set faithfulness")
        if i >= n - ncol:
            ax.set_xlabel("$K$ (candidate edges)")
    fig.tight_layout()
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"saved {args.out}.pdf / .png")

    # ---- single-panel summary, per-behaviour colours (fig_transfer style) ----
    # One K serves every behaviour, so the binding constraint is the worst (lowest) curve:
    # it reaches the 0.9 ceiling around K=kstar and flattens past it.
    plt.rcParams.update({"font.size": 13, "font.family": "sans-serif",
                         "axes.edgecolor": "#2b2b2b"})
    fig2, ax = plt.subplots(figsize=(9.4, 4.8))
    for name in names:
        attr = "kl-attr" if name in KL_TASKS else "logitdiff"
        cur = tasks[name].get(attr, {})
        ys = [cur.get(str(k)) for k in ks]
        ax.plot(ks, ys, "-o", color=COLORS.get(name, "#888"), lw=1.5, ms=3.5,
                alpha=0.95, label=LABELS.get(name, name), zorder=3)
    ax.axhline(0.9, color="#999", ls="--", lw=1, zorder=1)
    ax.text(max(ks) + 90, 0.9, "0.9", color="#999", va="center", fontsize=9)
    ax.axvline(args.kstar, color="#2b2b2b", ls=":", lw=1.5, zorder=2)
    ax.text(args.kstar, 0.05, f"chosen $K={args.kstar}$", color="#2b2b2b", ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), zorder=5)
    ax.set_xlabel("$K$ (candidate edges)", fontsize=13.5, labelpad=6)
    ax.set_ylabel("candidate-set faithfulness", fontsize=13.5, labelpad=6)
    ax.set_ylim(0, 1.04); ax.set_xlim(0, max(ks) + 250)
    ax.grid(color="#cfcfcf", lw=0.8, ls=":", zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(title="behaviour", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    frameon=True, edgecolor="#000000", fontsize=10.5, title_fontsize=11.5,
                    labelspacing=0.6, borderpad=0.8)
    leg.get_frame().set_linewidth(1.1)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(f"{args.out}_summary.{ext}", dpi=150, bbox_inches="tight")
    print(f"saved {args.out}_summary.pdf / .png")


if __name__ == "__main__":
    main()
