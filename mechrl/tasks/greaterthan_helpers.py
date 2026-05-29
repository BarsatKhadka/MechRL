"""Helpers for the greater-than task — kept out of greaterthan.py to keep the
wrapper itself short and readable.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def load_gpt2_small(device: str = "cpu") -> HookedTransformer:
    """Load GPT-2 small configured for ACDC-style edge editing."""
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in model.cfg.to_dict():
        model.set_use_hook_mlp_in(True)
    return model


def get_valid_years(tokenizer, start: int = 1000, end: int = 2150) -> torch.Tensor:
    """Years whose GPT-2 tokenization is exactly [' XX', 'YY'] (two tokens).

    Hanna's get_valid_years had a BOS bug under modern HF tokenizers — calling
    tokenizer(years) auto-prepends BOS (50256). We use add_special_tokens=False
    here. Also drops the first/last year in each century to avoid edge effects
    (matches Hanna's original logic).
    """
    years = [" " + str(year) for year in range(start, end)]
    tokens = tokenizer(years, add_special_tokens=False)["input_ids"]
    detokenized = [tokenizer.convert_ids_to_tokens(toks) for toks in tokens]
    valid = torch.tensor(
        [(len(detok) == 2 and len(detok[1]) == 2) for detok in detokenized]
    )

    # Drop first valid year of each century (and the immediately preceding
    # valid year) — these are the boundary cases where model behavior is
    # less reliable. Mirrors Hanna's original filter.
    last_valid_index = None
    current_century = None
    for i, year in zip(range(len(valid)), range(start, end)):
        cent = year // 100
        if valid[i]:
            if current_century != cent:
                current_century = cent
                valid[i] = False
                if last_valid_index is not None:
                    valid[last_valid_index] = False
            last_valid_index = i
    if last_valid_index is not None:
        valid[last_valid_index] = False

    return torch.arange(start, end)[valid]


def get_yy_token_ids(tokenizer) -> List[int]:
    """Token id for each two-digit year suffix '00' through '99'.

    GPT-2's BPE encodes these as single tokens when they appear after a year
    like ' 18' — the standalone '00'-'99' strings each tokenize to one id.
    """
    yy_strings = [f"{i:02d}" for i in range(100)]
    ids = []
    for s in yy_strings:
        toks = tokenizer(s, add_special_tokens=False)["input_ids"]
        assert len(toks) == 1, f"Expected 1 token for {s!r}, got {toks}"
        ids.append(toks[0])
    return ids


def _sentence_token_count(noun: str, year: int, kind: str, tokenizer) -> int:
    century = year // 100
    if kind == "good":
        s = f"The {noun} lasted from the year {year} to the year {century}"
    else:
        s = f"The {noun} lasted from the year {century}01 to the year {century}"
    return len(tokenizer(s, add_special_tokens=False)["input_ids"])


def _filter_nouns_for_uniform_length(nouns, years, tokenizer):
    """Keep only nouns whose good and bad sentences always tokenize to the
    SAME length across all candidate years (uniformly).

    The bug: " 1701" sometimes tokenizes as one token, sometimes two, depending
    on century. Mixing such bad-sentences in one batch crashes the tokenizer.
    Fix: choose a target token count and keep only nouns whose every (year,
    kind) pair hits that count.
    """
    # Probe with a small sample to learn the modal token count
    sample_year = int(years[len(years) // 2].item())
    sample_noun = nouns[0]
    target_good = _sentence_token_count(sample_noun, sample_year, "good", tokenizer)
    target_bad = _sentence_token_count(sample_noun, sample_year, "bad", tokenizer)
    # The two should match for a healthy sample; if not, prefer the good count
    target = target_good if target_good == target_bad else target_good

    safe = []
    for noun in nouns:
        ok = True
        for y in years:
            y_int = int(y.item())
            if _sentence_token_count(noun, y_int, "good", tokenizer) != target:
                ok = False
                break
            if _sentence_token_count(noun, y_int, "bad", tokenizer) != target:
                ok = False
                break
        if ok:
            safe.append(noun)
    if not safe:
        raise RuntimeError(
            f"No nouns produced uniform length-{target} good/bad pairs across all years."
        )
    return safe


def build_year_metric(
    tokenizer, years_YY: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a metric closure for greater-than.

    Returns a function (logits) -> per-prompt scalar. The score is:
        prob_diff = sum(P[good_YY]) - sum(P[bad_YY])
    where good_YY are token ids for two-digit years > years_YY[i], and bad_YY
    are those <= years_YY[i].

    Negated so lower-is-better (ACDC convention) — a model that strongly
    prefers valid year continuations gives a large NEGATIVE metric value.
    """
    yy_ids = torch.tensor(get_yy_token_ids(tokenizer))  # [100]

    def metric(logits: torch.Tensor) -> torch.Tensor:
        # logits: [batch, seq_len, vocab]
        last_logits = logits[:, -1, :]  # [batch, vocab]
        probs = F.softmax(last_logits, dim=-1)
        yy_probs = probs[:, yy_ids]  # [batch, 100]

        n = yy_probs.shape[0]
        yy_index = torch.arange(100, device=yy_probs.device).unsqueeze(0).expand(n, -1)
        threshold = years_YY.to(yy_probs.device).unsqueeze(1)  # [batch, 1]
        good_mask = yy_index > threshold  # [batch, 100]

        good_prob = (yy_probs * good_mask).sum(dim=-1)
        bad_prob = (yy_probs * (~good_mask)).sum(dim=-1)
        # Negate so lower = better, matching ACDC's metric sign convention.
        return -(good_prob - bad_prob).mean()

    return metric


# Noun list — copied from ACDC's greaterthan/utils.py so we don't depend on
# ACDC's wrapper code (which had the BOS bug). Vetted to produce reasonable
# year-context prompts.
GREATERTHAN_NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign",
    "captivity", "case", "challenge", "chaos", "clash",
    "collaboration", "coma", "competition", "confrontation", "consequence",
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal",
    "decline", "decrease", "demonstrations", "development", "disagreement",
    "disorder", "dispute", "domination", "dynasty", "effect",
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion",
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness",
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey",
    "kingdom", "marriage", "modernization", "negotiation",
    "notoriety", "obstruction", "operation", "order", "outbreak",
    "outcome", "overhaul", "patrols", "pilgrimage", "plague",
    "plan", "practice", "process", "program", "progress",
    "project", "pursuit", "quest", "raids", "reforms",
    "reign", "relationship",
    "retaliation", "riot", "rise", "rivalry", "romance",
    "rule", "sanctions", "shift", "siege", "slump",
    "stature", "stint", "strikes", "study",
    "test", "testing", "tests", "therapy", "tour",
    "tradition", "treaty", "trial", "trip", "unemployment",
    "voyage", "warfare", "work",
]
