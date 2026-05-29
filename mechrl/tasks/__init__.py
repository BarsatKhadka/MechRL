from mechrl.tasks.base import Task, TaskBatch

# Base / canonical tasks
from mechrl.tasks.ioi import IOITask
from mechrl.tasks.greaterthan import GreaterThanTask
from mechrl.tasks.induction import InductionTask
from mechrl.tasks.docstring import DocstringTask  # attn-only-4l (not GPT-2)
from mechrl.tasks.docstring_gpt2 import DocstringGPT2Task  # GPT-2 small

# Held-out test tasks
from mechrl.tasks.copy_suppression import CopySuppressionTask
from mechrl.tasks.successor_heads import SuccessorHeadsTask

# Cross-model / synthetic
from mechrl.tasks.tracr import TracrReverseTask, TracrProportionTask

# Verified passing variants (Gate 1 + Gate 2 passed)
from mechrl.tasks.greaterthan_variants import (
    GreaterThanVariantTask,
    GreaterThanOriginal,
    GreaterThanReversed,
    GreaterThanBeganEnded,
    GreaterThanTookPlace,
    GREATERTHAN_VARIANT_CLASSES,
)
from mechrl.tasks.ioi_variants import (
    IOIVariantTask,
    IOIAfterOpener,
    IOINoPlaceObject,
    IOIFriendsFound,
    IOI_VARIANT_CLASSES,
)
from mechrl.tasks.docstring_variants import (
    DocstringVariantTask,
    DocstringGPT2Sphinx7Task,
    DocstringGPT2Google5Task,
    DocstringGPT2ClassSphinxTask,
    DocstringGPT2Numpy5Task,
    DOCSTRING_VARIANT_CLASSES,
)


# Convenience: confirmed-working training pool
# (all tasks that passed Gate 1 + Gate 2 verification on GPT-2 small)
TRAINING_TASK_CLASSES = [
    IOITask,                       # ABBA canonical (70%)
    IOIAfterOpener,                # 77%
    IOINoPlaceObject,              # 72%
    IOIFriendsFound,               # 61%
    GreaterThanOriginal,           # 102%
    GreaterThanReversed,           # 92%
    GreaterThanBeganEnded,         # 104%
    GreaterThanTookPlace,          # 98%
    DocstringGPT2Task,             # 88% (5 args, sphinx style)
    DocstringGPT2Sphinx7Task,      # 69% (7 args, sphinx style)
    DocstringGPT2Google5Task,      # 87% (5 args, google style)
    DocstringGPT2ClassSphinxTask,  # 88% (5 args, class method, sphinx)
    DocstringGPT2Numpy5Task,       # 99% (5 args, numpy style)
]


__all__ = [
    "Task",
    "TaskBatch",
    # Base tasks
    "IOITask",
    "GreaterThanTask",
    "InductionTask",
    "DocstringTask",
    "DocstringGPT2Task",
    "CopySuppressionTask",
    "SuccessorHeadsTask",
    "TracrReverseTask",
    "TracrProportionTask",
    # Variant base classes + named subclasses
    "GreaterThanVariantTask",
    "GreaterThanOriginal",
    "GreaterThanReversed",
    "GreaterThanBeganEnded",
    "GreaterThanTookPlace",
    "GREATERTHAN_VARIANT_CLASSES",
    "IOIVariantTask",
    "IOIAfterOpener",
    "IOINoPlaceObject",
    "IOIFriendsFound",
    "IOI_VARIANT_CLASSES",
    "DocstringVariantTask",
    "DocstringGPT2Sphinx7Task",
    "DocstringGPT2Google5Task",
    "DocstringGPT2ClassSphinxTask",
    "DocstringGPT2Numpy5Task",
    "DOCSTRING_VARIANT_CLASSES",
    # Combined training pool
    "TRAINING_TASK_CLASSES",
]
