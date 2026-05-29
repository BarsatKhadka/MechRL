"""IOI (Indirect Object Identification) task from Wang et al. 2022.

Wraps ACDC's `get_all_ioi_things`. Uses ABBA prompt structure on GPT-2 small
with ABC patching for corrupted prompts. Metric: logit difference between the
indirect object (IO) and subject (S) tokens at the final position.
"""

from __future__ import annotations

from acdc.ioi.utils import get_all_ioi_things

from mechrl.tasks.base import Task, TaskBatch


class IOITask(Task):
    name = "ioi"

    def __init__(self, num_examples: int = 64, device: str = "cpu", seed: int = 0):
        super().__init__(num_examples=num_examples, device=device, seed=seed)

    def eap_labels(self, batch):
        """IOI: (IO_id, S_id) per prompt. S extracted from prompt as first
        repeated token (the subject who appears twice in ABBA structure).
        """
        import torch
        n = batch.batch_size
        tokenizer = self.model.tokenizer
        io_ids = batch.correct_labels[:n].tolist()
        s_ids = []
        for i in range(n):
            prompt = self.model.to_string(batch.clean_tokens[i])
            toks = tokenizer.encode(prompt, add_special_tokens=False)
            seen, s_id = set(), None
            for t in toks:
                if t in seen:
                    s_id = t
                    break
                seen.add(t)
            s_ids.append(s_id if s_id is not None else io_ids[i])
        return torch.tensor(list(zip(io_ids, s_ids)), dtype=torch.long)

    def _build(self) -> None:
        things = get_all_ioi_things(
            num_examples=self.num_examples,
            device=self.device,
            metric_name="logit_diff",
        )

        self._model = things.tl_model

        self._validation = TaskBatch(
            clean_tokens=things.validation_data,
            corrupted_tokens=things.validation_patch_data,
            correct_labels=things.validation_labels,
            wrong_labels=None,  # baked into metric closure by ACDC
            metric=things.validation_metric,
            metadata={"source": "acdc.ioi.utils.get_all_ioi_things", "prompt_type": "ABBA"},
        )

        self._test = TaskBatch(
            clean_tokens=things.test_data,
            corrupted_tokens=things.test_patch_data,
            correct_labels=things.test_labels,
            wrong_labels=None,
            metric=things.test_metrics["logit_diff"],
            metadata={"source": "acdc.ioi.utils.get_all_ioi_things", "prompt_type": "ABBA"},
        )
