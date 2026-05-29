# Task: tracr_proportion

- **Model**: custom (n_layers=2, n_heads=1)
- **Batch size**: 20
- **Sequence length**: 4
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: 0.0000
- **Metric aggregate (corrupted)**: 0.1417

Metadata:
```json
{
  "source": "acdc.tracr_task.utils.get_all_tracr_things(proportion)",
  "model": "tracr-compiled tiny transformer (NOT GPT-2 small)",
  "task_description": "proportion of 'x' tokens at each position"
}
```

## Examples

Showing first 20 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     `tokens=[0, 0, 0, 1]`
- **CORRUPTED**: `tokens=[2, 2, 2, 1]`
- **Difference**: 3/4 tokens differ at positions [0, 1, 2]

### Example 1

- **CLEAN**:     `tokens=[1, 2, 1, 1]`
- **CORRUPTED**: `tokens=[2, 2, 1, 3]`
- **Difference**: 2/4 tokens differ at positions [0, 3]

### Example 2

- **CLEAN**:     `tokens=[2, 3, 3, 0]`
- **CORRUPTED**: `tokens=[0, 3, 2, 1]`
- **Difference**: 3/4 tokens differ at positions [0, 2, 3]

### Example 3

- **CLEAN**:     `tokens=[0, 3, 2, 1]`
- **CORRUPTED**: `tokens=[1, 1, 3, 0]`
- **Difference**: 4/4 tokens differ at positions [0, 1, 2, 3]

### Example 4

- **CLEAN**:     `tokens=[3, 3, 1, 0]`
- **CORRUPTED**: `tokens=[3, 3, 2, 2]`
- **Difference**: 2/4 tokens differ at positions [2, 3]

### Example 5

- **CLEAN**:     `tokens=[2, 0, 0, 0]`
- **CORRUPTED**: `tokens=[2, 2, 3, 1]`
- **Difference**: 3/4 tokens differ at positions [1, 2, 3]

### Example 6

- **CLEAN**:     `tokens=[2, 2, 2, 3]`
- **CORRUPTED**: `tokens=[2, 3, 3, 0]`
- **Difference**: 3/4 tokens differ at positions [1, 2, 3]

### Example 7

- **CLEAN**:     `tokens=[2, 2, 1, 3]`
- **CORRUPTED**: `tokens=[0, 1, 1, 3]`
- **Difference**: 2/4 tokens differ at positions [0, 1]

### Example 8

- **CLEAN**:     `tokens=[3, 1, 3, 0]`
- **CORRUPTED**: `tokens=[3, 1, 3, 0]`
- **Difference**: **WARNING: clean and corrupted are IDENTICAL — bug**

### Example 9

- **CLEAN**:     `tokens=[2, 2, 0, 3]`
- **CORRUPTED**: `tokens=[2, 2, 0, 3]`
- **Difference**: **WARNING: clean and corrupted are IDENTICAL — bug**

### Example 10

- **CLEAN**:     `tokens=[1, 1, 3, 0]`
- **CORRUPTED**: `tokens=[2, 0, 0, 0]`
- **Difference**: 3/4 tokens differ at positions [0, 1, 2]

### Example 11

- **CLEAN**:     `tokens=[0, 2, 3, 0]`
- **CORRUPTED**: `tokens=[3, 3, 1, 0]`
- **Difference**: 3/4 tokens differ at positions [0, 1, 2]

### Example 12

- **CLEAN**:     `tokens=[2, 2, 3, 3]`
- **CORRUPTED**: `tokens=[0, 2, 3, 0]`
- **Difference**: 2/4 tokens differ at positions [0, 3]

### Example 13

- **CLEAN**:     `tokens=[0, 1, 3, 2]`
- **CORRUPTED**: `tokens=[0, 1, 3, 2]`
- **Difference**: **WARNING: clean and corrupted are IDENTICAL — bug**

### Example 14

- **CLEAN**:     `tokens=[2, 0, 3, 1]`
- **CORRUPTED**: `tokens=[0, 0, 0, 1]`
- **Difference**: 2/4 tokens differ at positions [0, 2]

### Example 15

- **CLEAN**:     `tokens=[3, 0, 3, 0]`
- **CORRUPTED**: `tokens=[2, 2, 3, 3]`
- **Difference**: 3/4 tokens differ at positions [0, 1, 3]

### Example 16

- **CLEAN**:     `tokens=[3, 3, 2, 2]`
- **CORRUPTED**: `tokens=[1, 2, 1, 1]`
- **Difference**: 4/4 tokens differ at positions [0, 1, 2, 3]

### Example 17

- **CLEAN**:     `tokens=[0, 1, 1, 3]`
- **CORRUPTED**: `tokens=[2, 0, 3, 1]`
- **Difference**: 4/4 tokens differ at positions [0, 1, 2, 3]

### Example 18

- **CLEAN**:     `tokens=[2, 2, 3, 1]`
- **CORRUPTED**: `tokens=[3, 0, 3, 0]`
- **Difference**: 3/4 tokens differ at positions [0, 1, 3]

### Example 19

- **CLEAN**:     `tokens=[2, 2, 2, 1]`
- **CORRUPTED**: `tokens=[2, 2, 2, 3]`
- **Difference**: 1/4 tokens differ at positions [3]
