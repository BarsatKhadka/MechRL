"""Share ONE frozen GPT-2 across many task bundles.

Each ACDC task family (ioi / greaterthan / docstring / ...) builds its own GPT-2
inside its `get_all_*_things` loader via `HookedTransformer.from_pretrained("gpt2")`
(see e.g. acdc/ioi/utils.py:get_gpt2_small). With N tasks that's N identical
copies of the model -> OOM on the 8GB 3070. But the model is identical and frozen
for every task -- the only thing EAP requires is the hook-flag config
(use_attn_result / use_split_qkv_input / use_hook_mlp_in), which is
task-INDEPENDENT.

So: build ONE model, and while the bundles are being constructed, intercept
`from_pretrained("gpt2")` so every loader gets handed back that same instance.
No transient copies are ever loaded -> nothing to free, and no risk of a lingering
reference (a metric closure, an AllDataThings field) silently keeping N models
resident. The flag-setters the loaders call afterwards are idempotent on the
shared model. Non-"gpt2" loads (e.g. tracr) fall through to the real loader.

Numerically this changes nothing vs the per-task path: each task would have
gotten an identical get_gpt2_small model anyway. Single-task runs (and resumes)
are unaffected.

Usage:
    shared = build_shared_gpt2(device)
    with use_shared_gpt2(shared):
        bundles = [TaskBundle.build(cls(device=device), k=K) for cls in classes]
    assert all(b.task.model is shared for b in bundles)   # one model, N tasks
"""

from __future__ import annotations

import contextlib

from transformer_lens import HookedTransformer

from acdc.ioi.utils import get_gpt2_small


def build_shared_gpt2(device: str = "cpu") -> HookedTransformer:
    """Build the single GPT-2 every task will share, with the EAP hook config.

    Identical to what each task's loader would have produced on its own (we reuse
    ACDC's get_gpt2_small), so behaviour/numerics match the per-task path exactly.
    """
    return get_gpt2_small(device=device)


@contextlib.contextmanager
def use_shared_gpt2(model: HookedTransformer):
    """While active, `HookedTransformer.from_pretrained("gpt2")` returns `model`.

    Restores the original loader on exit (even if a build raises). Build the
    shared model BEFORE entering this context (otherwise the build itself would
    be intercepted before `model` exists).
    """
    # Save the raw classmethod object so we restore the descriptor, not a bound
    # method (assigning a bound method back would break the classmethod protocol).
    had_own = "from_pretrained" in HookedTransformer.__dict__
    orig_attr = HookedTransformer.__dict__.get("from_pretrained")
    orig_call = HookedTransformer.from_pretrained          # bound, for fall-through

    def patched(name, *args, **kwargs):
        if name == "gpt2":
            return model
        return orig_call(name, *args, **kwargs)

    HookedTransformer.from_pretrained = staticmethod(patched)
    try:
        yield model
    finally:
        if had_own:
            HookedTransformer.from_pretrained = orig_attr
        else:                                              # pragma: no cover
            del HookedTransformer.from_pretrained
