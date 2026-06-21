"""Consolidate head recovery from circuit_*.json dumps for manual verification.

Pure JSON, no torch, no model -- safe to run as a tiny CPU job. For each dumped circuit
it RE-COMPUTES (does not trust the stored field) which canonical heads/MLPs are present,
and prints canonical / recovered / MISSING side by side, then writes a CSV. Cross-check the
canonical column against the papers, and the recovered column against the circuit.

Usage (as a job, not on a login node):
    python -m scripts.collate_heads --runs runs --out head_recovery.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt_h(h):
    return f"L{h[0]}H{h[1]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs", help="root dir to search for circuit_*.json")
    p.add_argument("--out", default="head_recovery.csv")
    args = p.parse_args()

    files = sorted(Path(args.runs).rglob("circuit_*.json"))
    print(f"[heads] {len(files)} circuit dumps under {args.runs}\n")

    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue

        canon = [tuple(h) for h in d.get("canonical_heads", [])]
        present = {tuple(h) for h in d.get("heads_present", [])}
        recovered = [h for h in canon if h in present]      # recomputed, not the stored field
        missing = [h for h in canon if h not in present]

        canon_m = list(d.get("canonical_mlps", []))
        present_m = set(d.get("mlps_present", []))
        rec_m = [m for m in canon_m if m in present_m]
        miss_m = [m for m in canon_m if m not in present_m]

        task = d.get("task", f.stem)
        faith = d.get("faith")
        nedges = d.get("n_edges")
        print(f"=== {task} ===  faith {faith}  |C| {nedges}  ({f.parent.name})")
        print(f"  canonical heads ({len(canon)}): {[fmt_h(h) for h in canon]}")
        print(f"  recovered ({len(recovered)}/{len(canon)}): {[fmt_h(h) for h in recovered]}")
        print(f"  MISSING ({len(missing)}): {[fmt_h(h) for h in missing]}")
        if canon_m:
            print(f"  canonical MLPs ({len(canon_m)}): {canon_m}  recovered {rec_m}  missing {miss_m}")
        print(f"  heads present in circuit ({len(present)} total): {sorted(fmt_h(h) for h in present)}")
        print()

        rows.append({
            "run": f.parent.name,
            "task": task,
            "faith": faith,
            "n_edges": nedges,
            "n_canonical": len(canon),
            "n_recovered": len(recovered),
            "recovery": f"{len(recovered)}/{len(canon)}",
            "canonical": " ".join(fmt_h(h) for h in canon),
            "recovered": " ".join(fmt_h(h) for h in recovered),
            "missing": " ".join(fmt_h(h) for h in missing),
            "mlp_canonical": " ".join(f"MLP{m}" for m in canon_m),
            "mlp_recovered": " ".join(f"MLP{m}" for m in rec_m),
            "heads_present": " ".join(sorted(fmt_h(h) for h in present)),
        })

    if rows:
        out = Path(args.out)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[heads] csv -> {out}  ({len(rows)} circuits)")
    else:
        print("[heads] no circuit_*.json found (run the dump job first)")


if __name__ == "__main__":
    main()
