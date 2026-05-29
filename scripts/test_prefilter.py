"""Test the EAP-IG prefilter on IOI.

Runs EAP-IG attribution on a batch of IOI prompts, then checks how many of
Wang et al.'s 26 canonical heads appear in the top-K edges at different K
values. This is the canonical-head retention test.

If top-K=3000 retains all (or nearly all) canonical heads, the prefilter is
working and we can lock K. If not, we either increase K or revisit the
method (vanilla EAP vs EAP-IG vs EAP-IG-activations).

Run from repo root:
    venv\\Scripts\\python.exe scripts\\test_prefilter.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial

from eap.attribute import attribute
from eap.graph import Graph

from mechrl.tasks import IOITask


# --- Wang et al. 2022 canonical IOI heads (Table 2) ---
# Format: (layer, head)
CANONICAL_IOI_HEADS: Set[Tuple[int, int]] = {
    # Name Movers
    (9, 6), (9, 9), (10, 0),
    # Backup Name Movers
    (10, 10), (10, 6), (10, 2), (9, 0), (9, 7),
    # Negative Name Movers
    (10, 7), (11, 10),
    # S-Inhibition Heads
    (7, 3), (7, 9), (8, 6), (8, 10),
    # Induction Heads
    (5, 5), (5, 8), (5, 9), (6, 9),
    # Duplicate Token Heads
    (0, 1), (0, 10), (3, 0),
    # Previous Token Heads
    (2, 2), (2, 9), (4, 11),
}


# ----- EAP-IG library expects strings + label tensors -----

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    return list(clean), list(corrupted), torch.stack(labels)


class IOIEapDataset(Dataset):
    """Wrap IOITask into the (clean_str, corrupted_str, label_tensor) format
    that eap-ig's dataloader expects.

    Label is [correct_token_id, wrong_token_id] for logit-diff.
    """

    def __init__(self, ioi_task: IOITask, split: str = "validation"):
        batch = ioi_task.validation_batch() if split == "validation" else ioi_task.test_batch()
        model = ioi_task.model
        # Decode the tokenized prompts back to strings (eap-ig re-tokenizes internally)
        self.clean_strs = [model.to_string(batch.clean_tokens[i]) for i in range(batch.batch_size)]
        self.corrupt_strs = [model.to_string(batch.corrupted_tokens[i]) for i in range(batch.batch_size)]

        # IOI labels: ACDC's IOIDataset gave us the correct (IO) token id in validation_labels.
        # We need a wrong distractor too — derive it from the prompt: the subject name (S).
        # The S name appears twice (first as A and again as the giver). We can extract it
        # from the original IOI dataset object.
        # For simplicity, we use the SECOND-occurring name token in the clean prompt:
        # in ABBA prompts ("Then, A and B went... A gave..."), the subject is A,
        # which appears at positions 2 and ~10.
        # We'll pull both labels from the underlying IOI dataset, which stored them.
        # Workaround: re-extract from prompts.
        io_ids = batch.correct_labels.tolist()  # the IO (Mary)
        # For S (wrong label), find the subject — first repeated name in prompt
        s_ids = []
        tokenizer = model.tokenizer
        for prompt in self.clean_strs:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            # Subject (S) appears twice; find first repeat
            seen, s_id = {}, None
            for tok in tokens:
                if tok in seen:
                    s_id = tok
                    break
                seen[tok] = True
            if s_id is None:
                s_id = io_ids[len(s_ids)]  # fallback to IO if no repeat
            s_ids.append(s_id)

        self.labels = torch.tensor(list(zip(io_ids, s_ids)), dtype=torch.long)

    def __len__(self):
        return len(self.clean_strs)

    def __getitem__(self, i):
        return self.clean_strs[i], self.corrupt_strs[i], self.labels[i]


def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean: bool = True,
    loss: bool = False,
):
    """Standard IOI metric: logit[IO] - logit[S] at the final position.

    Following eap-ig conventions: when loss=True, negate so lower is better.
    """
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    last_logits = logits[idx, input_length - 1]  # [batch, vocab]
    labels = labels.to(last_logits.device)
    good_bad = torch.gather(last_logits, -1, labels)  # [batch, 2]
    diff = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        diff = -diff
    if mean:
        diff = diff.mean()
    return diff


def heads_in_top_k(graph: Graph, k: int) -> Set[Tuple[int, int]]:
    """After attribution, return the set of (layer, head) involved in the top-K edges by score."""
    # Flatten all edges with their scores
    edge_list = list(graph.edges.values())
    # Sort by absolute score descending
    edge_list.sort(key=lambda e: abs(e.score.item() if torch.is_tensor(e.score) else e.score), reverse=True)
    top = edge_list[:k]

    heads = set()
    for edge in top:
        for node in (edge.parent, edge.child):
            # Parse "a{layer}.h{head}" → (layer, head)
            name = node.name
            if name.startswith("a"):
                try:
                    layer_part, head_part = name.split(".")
                    layer = int(layer_part[1:])
                    head = int(head_part[1:])
                    heads.add((layer, head))
                except (ValueError, IndexError):
                    pass
    return heads


def main():
    print("Loading IOI task...")
    ioi = IOITask(num_examples=50, device="cpu")
    model = ioi.model

    print("\nBuilding EAP-IG graph from model...")
    graph = Graph.from_model(model)
    print(f"  total edges in graph: {len(graph.edges)}")
    print(f"  total nodes in graph: {len(graph.nodes)}")

    print("\nWrapping IOI dataset for eap-ig...")
    ds = IOIEapDataset(ioi, split="validation")
    dataloader = DataLoader(ds, batch_size=10, collate_fn=collate_EAP)
    print(f"  {len(ds)} examples loaded")

    print("\nRunning EAP-IG attribution (this is the prefilter, ~10-30s on CPU)...")
    attribute(
        model,
        graph,
        dataloader,
        partial(logit_diff, loss=True, mean=True),
        method="EAP-IG-inputs",
        ig_steps=5,
    )

    print("\nCanonical-head retention test:")
    print(f"  Wang et al. canonical heads: {len(CANONICAL_IOI_HEADS)} heads")
    print()
    print(f"  {'K':>6} | {'heads in top-K':>14} | {'canonical retained':>22} | {'fraction':>10}")
    print(f"  {'-'*6} | {'-'*14} | {'-'*22} | {'-'*10}")
    for k in [500, 1000, 2000, 3000, 5000, 10000]:
        heads = heads_in_top_k(graph, k)
        retained = CANONICAL_IOI_HEADS & heads
        frac = len(retained) / len(CANONICAL_IOI_HEADS)
        print(f"  {k:>6} | {len(heads):>14} | {len(retained):>3}/{len(CANONICAL_IOI_HEADS):<3} = {frac:>8.1%}     | {frac:>9.2%}")

    print("\nMissing canonical heads at K=3000:")
    heads_3k = heads_in_top_k(graph, 3000)
    missing = CANONICAL_IOI_HEADS - heads_3k
    if not missing:
        print("  (none — all canonical heads retained!)")
    else:
        for layer, head in sorted(missing):
            print(f"  - L{layer}.H{head}")

    print("\nDone.")


if __name__ == "__main__":
    main()
