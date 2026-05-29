"""Induction task — matches Olsson et al. 2022 methodology.

Olsson's test: a sequence of RANDOM tokens [A][B][C]...[A] → predict [B].
The model must use the induction mechanism (look back for previous occurrence
of A, copy the token that came after it) — there's no real-text fallback.

Key design choices (changed from earlier "real text" version):
  - RANDOM tokens, not real English text (matches paper)
  - SHORT sequences (~15 tokens, not 49) — concentrates signal
  - Tokens sampled from a filtered "safe" vocab subset (avoids special tokens
    and rare tokens that GPT-2 can't predict reliably even with induction)

Sequence layout (half_len=8 example, total 15 tokens):
  [t0, t1, t2, t3, t4, t5, t6, t7,  t0, t1, t2, t3, t4, t5, t6]
   └────────── first half (8 tokens) ──┘   └──── second half (7) ───┘
                                                              ↑
                                                  predict t7 here

At the final position (last token = t6 from second half), the model should
predict t7 — because in the first half, t6 was followed by t7.

Corrupted version: same structure but with a DIFFERENT first half. The model
has no prior context to copy from, so induction can't work.

Metric: log P(correct_token) at the final position. Higher (less negative)
        means the model is more confident in the right answer.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# Tokens to avoid when sampling: special tokens, very rare tokens, etc.
# GPT-2's vocab is 50257. Token ids 0-50256.
# Special tokens: BOS/EOS is 50256. Avoid id 0 (sometimes special).
def _safe_token_pool(tokenizer, n_candidates: int = 2000) -> list[int]:
    """Return a list of token ids that roundtrip cleanly through
    decode → re-encode. This is critical because eap-ig converts our
    tokens to strings, then re-tokenizes them, and any tokens that
    re-tokenize differently cause shape mismatches downstream.

    Heuristic: prefer tokens with leading space (word boundaries),
    ASCII-only, of reasonable length. Then verify roundtrip.
    """
    pool = []
    for tok_id in range(1, 50256):
        try:
            s = tokenizer.decode([tok_id])
        except Exception:
            continue
        if not s or not s.strip():
            continue
        # Want tokens with leading space (natural word boundaries)
        if not s.startswith(" "):
            continue
        if len(s) < 3 or len(s) > 8:
            continue
        if not all(ord(c) < 128 for c in s):
            continue
        # Critical: must roundtrip to same single token
        try:
            reencoded = tokenizer.encode(s, add_special_tokens=False)
            if len(reencoded) != 1 or reencoded[0] != tok_id:
                continue
        except Exception:
            continue
        pool.append(tok_id)
        if len(pool) >= n_candidates:
            break
    if len(pool) < 100:
        raise RuntimeError(f"Token pool too small: only {len(pool)} survived roundtrip filter.")
    return pool


def _build_induction_batch(
    model: HookedTransformer,
    batch_size: int,
    half_len: int,
    seed: int,
    device: str,
):
    """Generate clean and corrupted induction prompts using random tokens.

    Clean:     [first_half | first_half[:-1]]  → predict first_half[-1]
    Corrupted: [other_half | first_half[:-1]]  → no prior context for the target
    """
    tokenizer = model.tokenizer
    pool = _safe_token_pool(tokenizer, n_candidates=2000)

    g = torch.Generator()
    g.manual_seed(seed)

    clean_seqs, corrupt_seqs, labels = [], [], []
    for _ in range(batch_size):
        # Sample two independent random sequences of length half_len
        idx_a = torch.randint(0, len(pool), (half_len,), generator=g)
        idx_b = torch.randint(0, len(pool), (half_len,), generator=g)
        first_half = torch.tensor([pool[i.item()] for i in idx_a], dtype=torch.long)
        other_half = torch.tensor([pool[i.item()] for i in idx_b], dtype=torch.long)

        # Ensure they're different (so corrupted actually corrupts)
        while torch.equal(first_half, other_half):
            idx_b = torch.randint(0, len(pool), (half_len,), generator=g)
            other_half = torch.tensor([pool[i.item()] for i in idx_b], dtype=torch.long)

        clean = torch.cat([first_half, first_half[:-1]])
        corrupt = torch.cat([other_half, first_half[:-1]])

        clean_seqs.append(clean)
        corrupt_seqs.append(corrupt)
        labels.append(first_half[-1].item())

    return (
        torch.stack(clean_seqs).to(device),
        torch.stack(corrupt_seqs).to(device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _build_induction_metric(labels: torch.Tensor) -> Callable:
    """logits -> mean log P(correct_token) at final position.
    Higher (less negative) is better.
    """

    def metric(logits: torch.Tensor) -> torch.Tensor:
        last_logits = logits[:, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)
        n = last_logits.shape[0]
        label_slice = labels[:n] if labels.shape[0] != n else labels
        chosen = log_probs.gather(1, label_slice.unsqueeze(1).to(log_probs.device)).squeeze(1)
        return chosen.mean()

    return metric


class InductionTask(Task):
    """Olsson-style induction on GPT-2 small with random tokens.

    Parameters
    ----------
    num_examples : number of prompts per split.
    half_len     : tokens in each "half." Total sequence is 2*half_len - 1.
                   Default 8 → 15-token sequence, matching Olsson's compact setup.
    """

    name = "induction"

    def __init__(
        self,
        num_examples: int = 64,
        half_len: int = 8,
        device: str = "cpu",
        seed: int = 0,
    ):
        super().__init__(num_examples=num_examples, device=device, seed=seed)
        self.half_len = half_len

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)

        clean_v, corrupt_v, labels_v = _build_induction_batch(
            model, self.num_examples, self.half_len, seed=self.seed, device=self.device
        )
        clean_t, corrupt_t, labels_t = _build_induction_batch(
            model, self.num_examples, self.half_len, seed=self.seed + 1, device=self.device
        )

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=clean_v,
            corrupted_tokens=corrupt_v,
            correct_labels=labels_v,
            wrong_labels=None,
            metric=_build_induction_metric(labels_v),
            metadata={
                "source": "Olsson et al. 2022 — random tokens, short [A][B]...[A] pattern",
                "half_len": self.half_len,
                "seq_len": clean_v.shape[1],
            },
        )
        self._test = TaskBatch(
            clean_tokens=clean_t,
            corrupted_tokens=corrupt_t,
            correct_labels=labels_t,
            wrong_labels=None,
            metric=_build_induction_metric(labels_t),
            metadata={
                "source": "Olsson et al. 2022 — random tokens, short [A][B]...[A] pattern",
                "half_len": self.half_len,
                "seq_len": clean_t.shape[1],
            },
        )
