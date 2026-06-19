"""Collate all saved result JSONs into the Results tables -- pure JSON parsing.

NO torch, NO transformers, NO model load -> safe to run on a login node. Reads the
per-task eval files written by zero_shot_eval (zeroshot_*, amort_*, validity_*) and the
ACDC baseline files (acdc_*), and prints markdown tables ready to drop into the paper.

Usage (login node is fine):
    python -m scripts.collate_results --runs runs
    python -m scripts.collate_results --runs runs --donor runs/train_seed1_1781368586
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:
        print(f"[skip] {p}: {e}")
        return None


def _g(d, k, default="-"):
    v = d.get(k, default)
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, list):
        return f"{len(v)}"
    return str(v)


def _n(v):
    return len(v) if isinstance(v, (list, tuple)) else v


def _heads(d):
    if "heads_total" in d:
        s = f"{_n(d.get('heads_recovered', []))}/{d['heads_total']}"
        if "mlps_total" in d:
            s += f" +MLP {_n(d.get('mlps_recovered', []))}/{d['mlps_total']}"
        return s
    return "-"


def eval_table(files, title):
    rows = []
    for f in sorted(files):
        try:
            d = _load(f)
            if d is None:
                continue
            rows.append((
                str(d.get("task", Path(f).stem)), _g(d, "faith"), _g(d, "n_edges"),
                _g(d, "argmax_accuracy"), _g(d, "logit_diff_recovery"),
                _g(d, "knockout_faith"), _g(d, "random_same_size_faith"), _heads(d),
            ))
        except Exception as e:
            print(f"[row-skip] {f}: {type(e).__name__}: {e}")
    print(f"\n### {title}  ({len(rows)} rows)\n")
    if not rows:
        print("(no rows)")
        return
    print("| task | faith | edges | argmax | logitdiff-rec | necessity | specificity | heads |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def acdc_table(files):
    rows = []
    for f in sorted(files):
        try:
            d = _load(f)
            if d is None:
                continue
            task = str(d.get("task", Path(f).stem))
            for r in d.get("results", []):
                rows.append((task, _g(r, "tau"), _g(r, "final_edges"),
                             _g(r, "final_faith"), _g(r, "forward_passes")))
        except Exception as e:
            print(f"[acdc-skip] {f}: {type(e).__name__}: {e}")
    print(f"\n### ACDC baseline (size / faith / cost)  ({len(rows)} rows)\n")
    if not rows:
        print("(no rows)")
        return
    print("| task | tau | edges | faith | forward-passes |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs", help="root dir holding run folders + acdc_*.json")
    p.add_argument("--donor", default=None, help="donor run dir (default: newest train_seed1_*)")
    args = p.parse_args()

    root = Path(args.runs)
    # Search the whole tree (rglob) so files are found regardless of which run dir
    # holds them -- the donor-dir-only globs missed amort/validity before.
    zs = sorted(root.rglob("zeroshot_*.json"))
    amort = sorted(root.rglob("amort_*.json"))
    valid = sorted(root.rglob("validity_*.json"))
    acdc = sorted(root.rglob("acdc_*.json"))
    print(f"[collate] runs={root}  found: zeroshot={len(zs)} amort={len(amort)} "
          f"validity={len(valid)} acdc={len(acdc)}")

    eval_table(zs, "5.1  Zero-shot transfer (frozen donor)")
    eval_table(amort, "5.2  Amortisation (12 training tasks, frozen replay)")
    eval_table(valid, "Warm-started circuit validity")
    acdc_table([f for f in acdc if "circuit_tau" not in f.name])  # skip per-tau circuit dumps

    print("\n[collate] done. (Gate tables -- ceilings, KL_cut, EAP-IG faith-vs-K -- are "
          "print-only; read them from the mechrl_*.out logs.)")


if __name__ == "__main__":
    main()
