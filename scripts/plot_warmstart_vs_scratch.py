"""Warm-start vs from-scratch learning curves on held-out behaviours (§5.1).

For each held-out task we have two single-task runs with identical bar (tau),
reward, candidate set and hyperparameters -- the ONLY difference is the policy
initialisation: from-scratch (random init, --init-from absent) vs warm-start
(--init-from the frozen 12-task donor). We plot faithfulness vs iteration for
both and read off iters-to-bar.

iters-to-bar is computed on a SMOOTHED curve (trailing window) and requires the
smoothed faith to STAY at/above tau (sustained crossing), not just touch it once
-- per-iter mean faith is stochastic, so a raw first-touch over-credits spikes.

Usage:
    python -m scripts.plot_warmstart_vs_scratch
    python -m scripts.plot_warmstart_vs_scratch --window 5 --out fig_warmscratch
"""
from __future__ import annotations
import argparse, glob, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Held-out behaviours shown in the transfer table, with display labels.
TASK_LABELS = {
    "GenderedPronounTask": "Gendered pronoun",
    "SubjectVerbAgreementTask": "Subject-verb",
    "AcronymTask": "Acronyms",
    "SimpleSyllogismTask": "Simple syllogism",
    "OppositeSyllogismTask": "Opposite syllogism",
    "CountryCapitalTask": "Country-capital",
    "MCQAnchoredBiasTask": "Multiple choice",
}


def load_run(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    lines = open(os.path.join(d, "metrics.jsonl")).read().splitlines()
    recs = [json.loads(l) for l in lines]
    iters = np.array([r["iter"] for r in recs], float)
    faith = np.array([r.get("faith", np.nan) for r in recs], float)
    kept = np.array([r.get("kept", np.nan) for r in recs], float)
    # tau lives in the single task's per_task block
    tau = None
    for r in recs[::-1]:
        pt = r.get("per_task", {})
        if pt:
            tau = next(iter(pt.values())).get("tau")
            break
    init = cfg.get("init_from")
    kind = "scratch" if init in (None, "", "null") else "warm"
    return {"dir": d, "task": cfg.get("tasks", ""), "kind": kind,
            "iters": iters, "faith": faith, "kept": kept, "tau": tau,
            "k": cfg.get("k")}


def smooth(x, w):
    if w <= 1:
        return x
    out = np.copy(x)
    for i in range(len(x)):
        out[i] = np.nanmean(x[max(0, i - w + 1): i + 1])  # trailing mean
    return out


def iters_to_bar(iters, faith, tau, w):
    """First iter at which the smoothed curve crosses tau AND stays >= tau after."""
    if tau is None:
        return None
    s = smooth(faith, w)
    above = s >= tau
    for i in range(len(above)):
        if above[i] and above[i:].all():
            return int(iters[i])
    return None


def pick_pairs(window):
    runs = [load_run(d) for d in glob.glob("runs/*/")
            if os.path.exists(os.path.join(d, "metrics.jsonl"))
            and os.path.exists(os.path.join(d, "config.json"))]
    pairs = {}
    for task in TASK_LABELS:
        cand = [r for r in runs if r["task"] == task]
        scr = [r for r in cand if r["kind"] == "scratch"]
        wrm = [r for r in cand if r["kind"] == "warm"]
        if not scr or not wrm:
            continue
        s = max(scr, key=lambda r: len(r["iters"]))           # the 200-iter scratch run
        # warm run whose tau matches the scratch bar (handles MCQ's two warm runs)
        wrm.sort(key=lambda r: (abs((r["tau"] or 0) - (s["tau"] or 0)), -len(r["iters"])))
        w = wrm[0]
        pairs[task] = (s, w)
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--out", default="figures/fig_warmscratch")
    args = p.parse_args()

    pairs = pick_pairs(args.window)

    # ---- summary: high faithfulness is easy (both reach it); minimality is the
    # discriminator -- warm-start reaches comparable faith with far fewer edges. ----
    print(f"{'behaviour':20} {'scr_f':6} {'wrm_f':6} {'scr_edges':9} {'wrm_edges':9} "
          f"{'edge_ratio':10} {'wrm@1-5':8}")
    ratios = []
    for task, (s, w) in pairs.items():
        sf, wf = np.nanmean(s["faith"][-5:]), np.nanmean(w["faith"][-5:])
        se, we = np.nanmean(s["kept"][-5:]), np.nanmean(w["kept"][-5:])
        we1 = np.nanmean(w["kept"][:5])           # warm edges in first 5 iters
        ratio = se / we if we else float("nan")
        ratios.append(ratio)
        print(f"{TASK_LABELS[task]:20} {sf:<6.3f} {wf:<6.3f} {se:<9.0f} {we:<9.0f} "
              f"{ratio:<10.2f} {we1:<8.0f}")
    print(f"\nedge-ratio (scratch/warm): median {np.nanmedian(ratios):.2f}x, "
          f"range {np.nanmin(ratios):.2f}-{np.nanmax(ratios):.2f}x")

    # ---- figure: fraction of candidate set retained vs iteration, PER behaviour ----
    # Normalised to |C|/K so all panels share a 0-1 axis (warm always low, scratch always
    # high reads at a glance); heavy smoothing turns the noisy trace into a trend. Sized
    # for full-page width (TMLR is single column, full text width).
    W = 11
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.4), sharex=True, sharey=True)
    for ax in axes.flat:
        ax.axis("off")
    for i, (task, (s, w)) in enumerate(pairs.items()):
        ax = axes.flat[i]; ax.axis("on")
        ssz = smooth(s["kept"] / s["k"], W); wsz = smooth(w["kept"] / w["k"], W)
        ax.fill_between(s["iters"], ssz, color="#c44", alpha=0.10)
        ax.fill_between(w["iters"], wsz, color="#258", alpha=0.10)
        ax.plot(s["iters"], ssz, color="#c44", lw=2.2, label="from scratch")
        ax.plot(w["iters"], wsz, color="#258", lw=2.2, label="warm start")
        # final faithfulness (matches Table 3), labelled on each line at its endpoint
        fw, fs = np.nanmean(w["faith"][-5:]), np.nanmean(s["faith"][-5:])
        xw, xs = float(w["iters"][-1]), float(s["iters"][-1])
        _bb = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7)
        # per-panel manual nudges (delta on the base 0.10 offset) to clear the curves
        _rdy = {"SubjectVerbAgreementTask": 0.16, "MCQAnchoredBiasTask": 0.08,
                "GenderedPronounTask": -0.06}.get(task, 0.0)
        _bdy = {"SubjectVerbAgreementTask": -0.08, "OppositeSyllogismTask": -0.08,
                "GenderedPronounTask": -0.06}.get(task, 0.0)
        _bdx = {"AcronymTask": 14}.get(task, 0.0)
        ax.text(xw + _bdx, wsz[-1] + 0.10 + _bdy, f"$f$={fw:.2f}", color="#258", bbox=_bb, zorder=5,
                fontsize=8.5, ha="center", va="bottom", fontweight="bold")
        ax.text(xs - 4, ssz[-1] + 0.10 + _rdy, f"$f$={fs:.2f}", color="#c44", bbox=_bb, zorder=5,
                fontsize=8.5, ha="right", va="bottom", fontweight="bold")
        ax.set_title(TASK_LABELS[task], fontsize=10.5)
        ax.set_ylim(0, 1.16); ax.set_xlim(0, 205)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.15)
        if i % 4 == 0: ax.set_ylabel("$|C|/K$ kept")
        if i >= 3: ax.set_xlabel("iteration")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[7].axis("off"); axes.flat[7].legend(h, l, loc="center", fontsize=11, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}_size.{ext}", dpi=150, bbox_inches="tight")
    print(f"saved {args.out}_size.pdf / .png")


if __name__ == "__main__":
    main()
