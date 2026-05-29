"""Smoke test: load GPT-2 small and build the ACDC computational graph.

If this runs end-to-end without error, the install is working and we can
start building the env on top.

Run from repo root:
    venv\Scripts\python.exe scripts\smoke_test.py
"""

import torch
from transformer_lens import HookedTransformer
from acdc.TLACDCCorrespondence import TLACDCCorrespondence


def main():
    print("Loading GPT-2 small (downloads ~500 MB on first run)...")
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    print(f"  n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
          f"d_model={model.cfg.d_model}")

    print("\nBuilding ACDC computational graph...")
    correspondence = TLACDCCorrespondence.setup_from_model(model, use_pos_embed=False)

    n_nodes = len(correspondence.nodes())
    n_edges = len(correspondence.all_edges())
    n_actionable = correspondence.count_no_edges()

    print(f"  nodes:                {n_nodes}")
    print(f"  total edges:          {n_edges}")
    print(f"  actionable edges:     {n_actionable}")
    print(f"  (placeholder edges:   {n_edges - n_actionable})")

    print("\nQuick forward pass on a toy prompt...")
    tokens = model.to_tokens("When John and Mary went to the store, John gave the bag to")
    with torch.no_grad():
        logits = model(tokens)
    print(f"  logits shape: {tuple(logits.shape)}")
    top_id = logits[0, -1].argmax().item()
    print(f"  top predicted next token: {model.to_string([top_id])!r}")

    print("\nSmoke test passed. Install is working.")


if __name__ == "__main__":
    main()
