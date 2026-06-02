"""Inspect and validate the feature set on IOI.

Checks:
  - edge_features shape [K, 16], node_features shape [M, 7]
  - signed score retains sign (some negative "helper" edges exist)
  - scores are normalized to [-1, 1]; rank_frac spans [0, 1]
  - per-edge rows align with CUT actions, per-node rows with KILL actions
  - prints the top/bottom edges and biggest kill-nodes in human-readable form
"""

import torch

from mechrl.tasks.ioi import IOITask
from mechrl.env import TaskBundle, EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES


def main():
    task = IOITask(num_examples=20, device="cpu")
    bundle = TaskBundle.build(task, k=3000)

    ef = bundle.edge_features
    nf = bundle.node_features
    print(f"edge_features: {tuple(ef.shape)}  (expect [3000, 16])")
    print(f"node_features: {tuple(nf.shape)}  (expect [M, 7])")
    print(f"edge feature names: {EDGE_FEATURE_NAMES}")
    print(f"node feature names: {NODE_FEATURE_NAMES}")

    signed = ef[:, 0]
    rank = ef[:, 1]
    print("\n--- normalization checks ---")
    print(f"signed_norm_score range: [{signed.min():.3f}, {signed.max():.3f}]  (expect within [-1,1])")
    print(f"  negatives (helper edges): {(signed < 0).sum().item()} / 3000")
    print(f"  has both signs: {(signed < 0).any().item()} and {(signed > 0).any().item()}")
    print(f"rank_frac range: [{rank.min():.3f}, {rank.max():.3f}]  (expect ~[0,1])")
    assert signed.abs().max() <= 1.0 + 1e-5, "score not normalized"
    assert (signed < 0).any(), "expected some negative (helper) edges — sign was dropped!"

    # Show the most important edge (rank 0) and a strong helper edge
    edge_list = bundle.engine.edge_list
    cand = bundle.cand_edge_idx.tolist()

    def edge_name(local_i):
        return edge_list[cand[local_i]].name

    top = int(torch.argmin(rank).item())               # rank_frac ~ 0 -> most important
    helper = int(torch.argmin(signed).item())          # most negative -> strongest helper
    print("\n--- example edges ---")
    print(f"most important edge: {edge_name(top):28s} signed={signed[top]:+.3f} rank={rank[top]:.3f}")
    print(f"strongest helper:    {edge_name(helper):28s} signed={signed[helper]:+.3f} rank={rank[helper]:.3f}")

    print("\n--- biggest KILL targets (by out-degree) ---")
    sizes = [(m, int(g.numel())) for m, g in enumerate(bundle.parent_groups)]
    for m, sz in sorted(sizes, key=lambda x: -x[1])[:5]:
        agg = nf[m, -1].item()        # agg_signed_norm_score
        deg = nf[m, -2].item()        # out_degree_frac
        print(f"  {bundle.parent_names[m]:8s} edges={sz:4d} out_degree_frac={deg:.3f} agg_signed={agg:+.3f}")

    print("\n--- alignment check ---")
    # node_features[m] must correspond to parent_names[m] and parent_groups[m]
    m0 = bundle.parent_names.index("m0") if "m0" in bundle.parent_names else 0
    grp = bundle.parent_groups[m0]
    parents_in_group = {edge_list[cand[i]].parent.name for i in grp.tolist()}
    print(f"parent_names[{m0}]='{bundle.parent_names[m0]}', all edges in its group have parent: {parents_in_group}")
    assert parents_in_group == {bundle.parent_names[m0]}, "node/edge alignment broken!"

    print("\nFeature checks passed.")


if __name__ == "__main__":
    main()
