"""Design and validate greater-than task variants.

For each candidate template:
  GATE 1: Does the full GPT-2 model solve this task well?
          (Hanna et al. report 81.7% probability difference on original)
  GATE 2: Does EAP-IG top-3000 preserve meaningful faithfulness?
          (Need >60% for the agent to have a learning target)

Only templates passing BOTH gates become training tasks.

Template constraint: must end with " {century}" so the model predicts YY.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import (
    GREATERTHAN_NOUNS,
    build_year_metric,
    get_valid_years,
    load_gpt2_small,
)

_HANNA_ROOT = Path(__file__).resolve().parents[1] / "paperCodes" / "gpt2-greater-than"
if str(_HANNA_ROOT) not in sys.path:
    sys.path.insert(0, str(_HANNA_ROOT))


# Candidate templates — all end with " {century}" so model predicts YY
TEMPLATES = {
    "original": (
        lambda noun, year, century: f"The {noun} lasted from the year {year} to the year {century}",
        lambda noun, year, century: f"The {noun} lasted from the year {century}01 to the year {century}",
    ),
    "reversed_from_to": (
        lambda noun, year, century: f"From the year {year} to the year {century}",
        lambda noun, year, century: f"From the year {century}01 to the year {century}",
    ),
    "began_ended": (
        lambda noun, year, century: f"The {noun} began in {year} and ended in {century}",
        lambda noun, year, century: f"The {noun} began in {century}01 and ended in {century}",
    ),
    "took_place_between": (
        lambda noun, year, century: f"The {noun} took place between {year} and {century}",
        lambda noun, year, century: f"The {noun} took place between {century}01 and {century}",
    ),
}


def compute_prob_diff(model, tokens, years_YY):
    """Compute Hanna's probability difference metric on a batch.
    Higher = better. Hanna reports 0.817 on full GPT-2 on the original task.
    """
    tokenizer = model.tokenizer
    yy_strings = [f"{i:02d}" for i in range(100)]
    yy_ids = []
    for s in yy_strings:
        toks = tokenizer(s, add_special_tokens=False)["input_ids"]
        assert len(toks) == 1
        yy_ids.append(toks[0])
    yy_ids = torch.tensor(yy_ids)

    with torch.no_grad():
        logits = model(tokens)
    last_logits = logits[:, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    yy_probs = probs[:, yy_ids]

    n = yy_probs.shape[0]
    yy_index = torch.arange(100).unsqueeze(0).expand(n, -1)
    threshold = years_YY.unsqueeze(1)
    good_mask = yy_index > threshold

    good_prob = (yy_probs * good_mask).sum(dim=-1)
    bad_prob = (yy_probs * (~good_mask)).sum(dim=-1)
    return (good_prob - bad_prob)  # per-prompt


def generate_batch(model, template_name, clean_fn, bad_fn, n_examples, seed):
    """Generate a batch using the given template, with the uniform-length filter."""
    tokenizer = model.tokenizer
    valid_years = get_valid_years(tokenizer)

    random.seed(seed)
    torch.manual_seed(seed)

    n_candidates = n_examples * 5
    nouns = random.choices(GREATERTHAN_NOUNS, k=n_candidates)
    year_idx = torch.randint(0, len(valid_years), (n_candidates,))
    years = valid_years[year_idx]

    good_ids_list, bad_ids_list, kept_years = [], [], []
    for noun, year in zip(nouns, years):
        y = int(year.item())
        century = y // 100
        good_str = clean_fn(noun, y, century)
        bad_str = bad_fn(noun, y, century)
        good_ids = tokenizer(good_str, add_special_tokens=False)["input_ids"]
        bad_ids = tokenizer(bad_str, add_special_tokens=False)["input_ids"]
        good_ids_list.append(good_ids)
        bad_ids_list.append(bad_ids)
        kept_years.append(y)

    paired_lengths = [(len(g), len(b)) for g, b in zip(good_ids_list, bad_ids_list)]
    length_counts = Counter(lg for (lg, lb) in paired_lengths if lg == lb)
    if not length_counts:
        return None, None, None, None
    target_len = length_counts.most_common(1)[0][0]

    kept = [
        (g, b, y)
        for g, b, y in zip(good_ids_list, bad_ids_list, kept_years)
        if len(g) == target_len and len(b) == target_len
    ]
    if len(kept) < n_examples:
        return None, None, None, None
    kept = kept[:n_examples]

    good_toks = torch.tensor([g for g, _, _ in kept], dtype=torch.long)
    bad_toks = torch.tensor([b for _, b, _ in kept], dtype=torch.long)
    years_YY = torch.tensor([y % 100 for _, _, y in kept], dtype=torch.long)
    return good_toks, bad_toks, years_YY, target_len


def main():
    print("Loading GPT-2 small...")
    model = load_gpt2_small(device="cpu")

    print(f"\n{'GATE 1: Full model probability difference per template':^70}")
    print(f"{'(Hanna reports 0.817 on original. Pass if >0.5)':^70}\n")
    print(f"  {'template':>25} | {'#prompts':>9} | {'seq_len':>7} | {'mean prob_diff':>14} | {'sample':>50}")
    print(f"  {'-'*25} | {'-'*9} | {'-'*7} | {'-'*14} | {'-'*50}")

    results = {}
    n_examples = 30
    for tname, (clean_fn, bad_fn) in TEMPLATES.items():
        good_toks, bad_toks, years_YY, seq_len = generate_batch(model, tname, clean_fn, bad_fn, n_examples, seed=0)
        if good_toks is None:
            print(f"  {tname:>25} | {'FAIL':>9} | {'-':>7} | {'tokenization issue':>14} | -")
            continue

        prob_diff = compute_prob_diff(model, good_toks, years_YY)
        mean = prob_diff.mean().item()

        sample = model.to_string(good_toks[0])
        if len(sample) > 50:
            sample = sample[:47] + "..."
        results[tname] = {
            "mean_prob_diff": mean,
            "good_toks": good_toks,
            "bad_toks": bad_toks,
            "years_YY": years_YY,
            "passes_gate1": mean > 0.5,
        }
        passes = "PASS" if mean > 0.5 else "FAIL"
        print(f"  {tname:>25} | {len(good_toks):>9} | {seq_len:>7} | {mean:>13.3f}  | {sample!r}")
        print(f"  {'':>25} | {'':>9} | {'':>7} | {'':>14} | [{passes}]")


if __name__ == "__main__":
    main()
