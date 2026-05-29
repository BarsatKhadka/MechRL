"""IOI task variants — different prompt templates from Wang et al.

Verified passing results (CPU run, K=3000 EAP-IG faithfulness):
  - original ABBA canonical : 70.14%  (use base IOITask for this)
  - after_opener            : 76.76%
  - no_place_object         : 71.94%
  - friends_found           : 60.96%

Note: ACDC's IOIDataset uses BABA-style templates with [A], [B] placeholders.
When passed as a literal template list (prompt_type=[template]), the [A] / [B]
fill order matters. We use ABBA-style names (subject is A, indirect object is B,
order in template is A then B then B).
"""

from __future__ import annotations

from functools import partial

import torch

from acdc.acdc_utils import logit_diff_metric
from acdc.ioi.ioi_dataset import IOIDataset

from mechrl.tasks.base import Task, TaskBatch
from mechrl.tasks.greaterthan_helpers import load_gpt2_small


# Template registry — name → ABBA-style template string
IOI_TEMPLATES = {
    "after_opener":     "After [A] and [B] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "no_place_object":  "Then, [A] and [B] had a long argument. Afterwards [B] said to [A]",
    "friends_found":    "Friends [A] and [B] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
}


class IOIVariantTask(Task):
    """Generic IOI variant using a specific template string from IOI_TEMPLATES."""

    def __init__(self, variant: str, num_examples: int = 30, device: str = "cpu", seed: int = 0):
        if variant not in IOI_TEMPLATES:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {list(IOI_TEMPLATES.keys())}"
            )
        super().__init__(num_examples=num_examples, device=device, seed=seed)
        self.variant = variant
        self.template = IOI_TEMPLATES[variant]
        self.name = f"ioi_{variant}"

    def _build(self):
        model = load_gpt2_small(device=self.device)

        ioi = IOIDataset(
            prompt_type=[self.template],
            N=self.num_examples * 2,
            nb_templates=1,
            seed=self.seed,
        )
        abc = (
            ioi.gen_flipped_prompts(("IO", "RAND"), seed=self.seed + 100)
               .gen_flipped_prompts(("S", "RAND"), seed=self.seed + 200)
               .gen_flipped_prompts(("S1", "RAND"), seed=self.seed + 300)
        )

        seq_len = ioi.toks.shape[1]
        n = self.num_examples

        val_clean = ioi.toks.long()[:n, : seq_len - 1].to(self.device)
        val_corr = abc.toks.long()[:n, : seq_len - 1].to(self.device)
        val_io = ioi.toks.long()[:n, seq_len - 1].to(self.device)
        val_s = torch.as_tensor(ioi.s_tokenIDs[:n], dtype=torch.long, device=self.device)

        test_clean = ioi.toks.long()[n:, : seq_len - 1].to(self.device)
        test_corr = abc.toks.long()[n:, : seq_len - 1].to(self.device)
        test_io = ioi.toks.long()[n:, seq_len - 1].to(self.device)
        test_s = torch.as_tensor(ioi.s_tokenIDs[n:], dtype=torch.long, device=self.device)

        val_metric = partial(logit_diff_metric, correct_labels=val_io, wrong_labels=val_s)
        test_metric = partial(logit_diff_metric, correct_labels=test_io, wrong_labels=test_s)

        self._model = model
        self._validation = TaskBatch(
            clean_tokens=val_clean, corrupted_tokens=val_corr,
            correct_labels=val_io, wrong_labels=val_s,
            metric=val_metric,
            metadata={"task": "ioi", "variant": self.variant, "template": self.template},
        )
        self._test = TaskBatch(
            clean_tokens=test_clean, corrupted_tokens=test_corr,
            correct_labels=test_io, wrong_labels=test_s,
            metric=test_metric,
            metadata={"task": "ioi", "variant": self.variant, "template": self.template},
        )

    def eap_labels(self, batch):
        n = batch.batch_size
        return torch.stack([batch.correct_labels[:n], batch.wrong_labels[:n]], dim=1)


# Convenience subclasses
class IOIAfterOpener(IOIVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="after_opener", **kwargs)

class IOINoPlaceObject(IOIVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="no_place_object", **kwargs)

class IOIFriendsFound(IOIVariantTask):
    def __init__(self, **kwargs):
        super().__init__(variant="friends_found", **kwargs)


# Use this list + the base IOITask (which is the ABBA canonical) for training
IOI_VARIANT_CLASSES = [
    IOIAfterOpener,
    IOINoPlaceObject,
    IOIFriendsFound,
]
