"""Verify KL faithfulness behaves correctly on IOI.

The decisive check: under KL, cutting the negative name-mover a10.h7 should
LOWER faith (it makes the circuit diverge from the model), whereas under
logit-diff it RAISED faith (0.72 -> 0.96). That flip is the whole point — KL
keeps the suppressors instead of rewarding their removal.
"""

import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import build_graph, Prefilter, AblationEngine


def main():
    print("Loading IOI...")
    task = IOITask(num_examples=20, device="cpu")
    graph = build_graph(task.model)

    eng_kl = AblationEngine(task, graph, metric_type="kl")
    eng_ld = AblationEngine(task, graph, metric_type=None)  # logit-diff (old)

    pref = Prefilter(task, graph, ig_steps=5)
    pref.compute(batch_size=10)
    cand = pref.candidate_mask(3000)

    print("\n--- baselines ---")
    print(f"KL  full_baseline (KL of full vs full, ~0):  {eng_kl.full_baseline():.5f}")
    print(f"KL  corrupted_baseline (KL of all-cut):      {eng_kl.corrupted_baseline():.5f}")

    print("\n--- faithfulness of top-3k ---")
    f_kl = eng_kl.faithfulness(cand)
    f_ld = eng_ld.faithfulness(cand)
    print(f"  KL-faith   top-3k: {f_kl:.4f}   (caps at 1.0)")
    print(f"  logit-diff top-3k: {f_ld:.4f}   (old metric, for reference)")

    # find the a10.h7 -> logits edge (negative name mover / suppressor)
    edge_list = eng_kl.edge_list
    cand_idx = cand.nonzero(as_tuple=True)[0]
    target = None
    for i in cand_idx.tolist():
        if edge_list[i].name == "a10.h7->logits":
            target = i
            break
    assert target is not None, "a10.h7->logits not in candidates?"

    mask_no_supp = cand.clone()
    mask_no_supp[target] = False

    print("\n--- THE decisive check: cut the suppressor a10.h7->logits ---")
    kl_before, kl_after = eng_kl.faithfulness(cand), eng_kl.faithfulness(mask_no_supp)
    ld_before, ld_after = eng_ld.faithfulness(cand), eng_ld.faithfulness(mask_no_supp)
    print(f"  KL-faith   : {kl_before:.4f} -> {kl_after:.4f}   (KL should DROP -> keep suppressor)")
    print(f"  logit-diff : {ld_before:.4f} -> {ld_after:.4f}   (logit-diff RISES -> cut suppressor)")

    if kl_after < kl_before:
        print("\nPASS: under KL, cutting the suppressor LOWERS faith -> it gets kept. (opposite of logit-diff)")
    else:
        print("\nUNEXPECTED: KL did not drop on cutting the suppressor — investigate.")

    # sanity: faith never exceeds 1.0 under KL
    print(f"\nfaith cap check: KL-faith of full candidate set = {f_kl:.4f} (must be <= ~1.0)")


if __name__ == "__main__":
    main()
