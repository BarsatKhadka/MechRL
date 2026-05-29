# Task: tracr_reverse

- **Model**: custom (n_layers=4, n_heads=1)
- **Batch size**: 6
- **Sequence length**: 4
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: 0.0000
- **Metric aggregate (corrupted)**: 0.4444

Metadata:
```json
{
  "source": "acdc.tracr_task.utils.get_all_tracr_things(reverse)",
  "model": "tracr-compiled tiny transformer (NOT GPT-2 small)",
  "task_description": "reverse [a, b, c] -> [c, b, a]"
}
```

## Examples

Showing first 6 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     `tokens=[3, 0, 1, 2]`
- **CORRUPTED**: `tokens=[3, 0, 2, 1]`
- **Difference**: 2/4 tokens differ at positions [2, 3]

### Example 1

- **CLEAN**:     `tokens=[3, 0, 2, 1]`
- **CORRUPTED**: `tokens=[3, 2, 0, 1]`
- **Difference**: 2/4 tokens differ at positions [1, 2]

### Example 2

- **CLEAN**:     `tokens=[3, 1, 0, 2]`
- **CORRUPTED**: `tokens=[3, 0, 1, 2]`
- **Difference**: 2/4 tokens differ at positions [1, 2]

### Example 3

- **CLEAN**:     `tokens=[3, 1, 2, 0]`
- **CORRUPTED**: `tokens=[3, 2, 1, 0]`
- **Difference**: 2/4 tokens differ at positions [1, 2]

### Example 4

- **CLEAN**:     `tokens=[3, 2, 0, 1]`
- **CORRUPTED**: `tokens=[3, 1, 0, 2]`
- **Difference**: 2/4 tokens differ at positions [1, 3]

### Example 5

- **CLEAN**:     `tokens=[3, 2, 1, 0]`
- **CORRUPTED**: `tokens=[3, 1, 2, 0]`
- **Difference**: 2/4 tokens differ at positions [1, 2]
