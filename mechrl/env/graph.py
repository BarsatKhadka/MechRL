"""Graph construction — wraps EAP-IG's Graph.from_model.

The graph represents GPT-2 small's computational structure:
- 158 nodes (144 attention heads + 12 MLPs + input + logits)
- ~32,491 edges (every (sender, receiver, channel) tuple respecting causality)

This is task-independent — same graph for all tasks running on the same model.
"""

from __future__ import annotations

from transformer_lens import HookedTransformer
from eap.graph import Graph


def build_graph(model: HookedTransformer) -> Graph:
    """Build the full computational graph for the given model.

    Also ensures the model is configured with the hooks EAP-IG needs:
    use_attn_result, use_split_qkv_input, use_hook_mlp_in.
    """
    if not model.cfg.use_attn_result:
        model.set_use_attn_result(True)
    if not model.cfg.use_split_qkv_input:
        model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in model.cfg.to_dict() and not model.cfg.use_hook_mlp_in:
        model.set_use_hook_mlp_in(True)

    graph = Graph.from_model(model)
    return graph
