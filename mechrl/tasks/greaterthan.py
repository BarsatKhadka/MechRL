"""Greater-than task from Hanna, Liu, Variengien (2023).

We use Hanna's sentence-construction helpers (generate_real_sentence,
generate_bad_sentence) but do tokenization ourselves to handle a corner case:
under modern HF tokenizers, the same template can tokenize to different lengths
depending on the noun and the year, which breaks batched tokenization. We
generate candidates, tokenize each individually, and keep only those that hit
the modal token count.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import torch

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import (
    GREATERTHAN_NOUNS,
    build_year_metric,
    get_valid_years,
    load_gpt2_small,
)

_HANNA_ROOT = Path(__file__).resolve().parents[2] / "paperCodes" / "gpt2-greater-than"
if str(_HANNA_ROOT) not in sys.path:
    sys.path.insert(0, str(_HANNA_ROOT))

from dataset import generate_real_sentence, generate_bad_sentence  # noqa: E402


class GreaterThanTask(Task):
    name = "greaterthan"

    def __init__(self, num_examples: int = 64, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def _build(self) -> None:
        model = load_gpt2_small(device=self.device)
        tokenizer = model.tokenizer

        valid_years = get_valid_years(tokenizer)
        n_total = self.num_examples * 2  # validation + test

        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Oversample (5x) then filter to those with the modal token length so
        # we end up with n_total uniform-length pairs. 5x is usually enough.
        oversample_factor = 5
        n_candidates = n_total * oversample_factor

        nouns = random.choices(GREATERTHAN_NOUNS, k=n_candidates)
        year_idx = torch.randint(0, len(valid_years), (n_candidates,))
        years = valid_years[year_idx]

        good_ids_list, bad_ids_list, kept_years = [], [], []
        for noun, year in zip(nouns, years):
            y = int(year.item())
            good_str = generate_real_sentence(noun, y, eos=False)
            bad_str = generate_bad_sentence(noun, y, eos=False)
            good_ids = tokenizer(good_str, add_special_tokens=False)["input_ids"]
            bad_ids = tokenizer(bad_str, add_special_tokens=False)["input_ids"]
            good_ids_list.append(good_ids)
            bad_ids_list.append(bad_ids)
            kept_years.append(y)

        # Find modal length where both halves agree, keep only those examples
        paired_lengths = [
            (len(g), len(b)) for g, b in zip(good_ids_list, bad_ids_list)
        ]
        length_counts = Counter(
            lg for (lg, lb) in paired_lengths if lg == lb
        )
        if not length_counts:
            raise RuntimeError("No (good, bad) pairs with matching length found.")
        target_len = length_counts.most_common(1)[0][0]

        kept = [
            (g, b, y)
            for g, b, y in zip(good_ids_list, bad_ids_list, kept_years)
            if len(g) == target_len and len(b) == target_len
        ]
        if len(kept) < n_total:
            raise RuntimeError(
                f"Only got {len(kept)} uniform-length pairs, need {n_total}. "
                f"Increase oversample_factor."
            )
        kept = kept[:n_total]

        good_toks = torch.tensor([g for g, _, _ in kept], dtype=torch.long, device=self.device)
        bad_toks = torch.tensor([b for _, b, _ in kept], dtype=torch.long, device=self.device)
        years_YY = torch.tensor([y % 100 for _, _, y in kept], dtype=torch.long, device=self.device)
        years_full = torch.tensor([y for _, _, y in kept], dtype=torch.long, device=self.device)

        n = self.num_examples
        val_metric = build_year_metric(tokenizer, years_YY[:n])
        test_metric = build_year_metric(tokenizer, years_YY[n:])

        # Decode a sample for the metadata
        sample_clean = tokenizer.decode(good_toks[0].tolist())
        sample_corrupt = tokenizer.decode(bad_toks[0].tolist())

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=good_toks[:n],
            corrupted_tokens=bad_toks[:n],
            correct_labels=years_YY[:n],
            wrong_labels=None,
            metric=val_metric,
            metadata={
                "source": "Hanna et al. 2023 (helpers from paperCodes/gpt2-greater-than)",
                "prompt_example": sample_clean,
                "corrupted_example": sample_corrupt,
                "full_years": years_full[:n].tolist(),
                "target_token_length": target_len,
            },
        )
        self._test = TaskBatch(
            clean_tokens=good_toks[n:],
            corrupted_tokens=bad_toks[n:],
            correct_labels=years_YY[n:],
            wrong_labels=None,
            metric=test_metric,
            metadata={
                "source": "Hanna et al. 2023 (helpers from paperCodes/gpt2-greater-than)",
                "prompt_example": tokenizer.decode(good_toks[n].tolist()),
                "target_token_length": target_len,
            },
        )
