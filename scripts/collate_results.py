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


def _heads(d):
    if "heads_total" in d:
        rec = len(d.get("heads_recovered", []))
        s = f"{rec}/{d['heads_total']}"
        if "mlps_total" in d:
            s += f" +MLP {len(d.get('mlps_recovered', []))}/{d['mlps_total']}"
        return s
    return "-"


def eval_table(files, title):
    rows = []
    for f in sorted(files):
        d = _load(f)
        if d is None:
            continue
        rows.append((
            d.get("task", Path(f).stem), _g(d, "faith"), _g(d, "n_edges"),
            _g(d, "argmax_accuracy"), _g(d, "logit_diff_recovery"),
            _g(d, "knockout_faith"), _g(d, "random_same_size_faith"), _heads(d),
        ))
    if not rows:
        return
    print(f"\n### {title}\n")
    print("| task | faith | edges | argmax | logitdiff-rec | necessity | specificity | heads |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def acdc_table(files):
    rows = []
    for f in sorted(files):
        d = _load(f)
        if d is None:
            continue
        task = d.get("task", Path(f).stem)
        for r in d.get("results", []):
            rows.append((task, _g(r, "tau"), _g(r, "final_edges"),
                         _g(r, "final_faith"), _g(r, "forward_passes")))
    if not rows:
        return
    print("\n### ACDC baseline (size / faith / cost)\n")
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
    donor = Path(args.donor) if args.donor else None
    if donor is None:
        cands = sorted(root.glob("train_seed1_*"), key=lambda q: q.name)
        donor = cands[-1] if cands else root
    print(f"[collate] runs={root}  donor={donor}")

    # §5.1 zero-shot transfer (held-out) + frozen-replay sanity, from the donor dir.
    eval_table(list(donor.glob("zeroshot_*.json")), "5.1  Zero-shot transfer (frozen donor)")
    # §5.2 amortisation: donor replayed on its 12 training tasks.
    eval_table(list(donor.glob("amort_*.json")), "5.2  Amortisation (12 training tasks, frozen replay)")
    # Warm-start validity, across all warm-started run dirs.
    eval_table(list(root.glob("*_seed1_*/validity_*.json")), "Warm-started circuit validity")
    # ACDC baseline.
    acdc_table(list(root.glob("acdc_*.json")))

    print("\n[collate] done. (Gate tables -- ceilings, KL_cut, EAP-IG faith-vs-K -- are "
          "print-only; read them from the mechrl_*.out logs.)")


if __name__ == "__main__":
    main()
