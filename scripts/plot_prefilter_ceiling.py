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


if __name__ == "__main__":
    main()
