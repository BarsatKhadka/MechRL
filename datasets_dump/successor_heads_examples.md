# Task: successor_heads

- **Model**: gpt2 (n_layers=12, n_heads=12)
- **Batch size**: 20
- **Sequence length**: 5
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: -0.3650
- **Metric aggregate (corrupted)**: 0.1561

Metadata:
```json
{
  "source": "Gould et al. 2024 (synthetic templated version)",
  "categories": [
    "days",
    "months",
    "numbers"
  ],
  "task": "predict next item in ordered sequence"
}
```

## Examples

Showing first 20 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     `The month after July is`
- **CORRUPTED**: `The month after January is`
- **Difference**: 1/5 tokens differ at positions 3; clean='July' vs corrupted='January'

### Example 1

- **CLEAN**:     `The month after September is`
- **CORRUPTED**: `The month after August is`
- **Difference**: 1/5 tokens differ at positions 3; clean='September' vs corrupted='August'

### Example 2

- **CLEAN**:     `The month after May is`
- **CORRUPTED**: `The month after September is`
- **Difference**: 1/5 tokens differ at positions 3; clean='May' vs corrupted='September'

### Example 3

- **CLEAN**:     `The month after October is`
- **CORRUPTED**: `The month after April is`
- **Difference**: 1/5 tokens differ at positions 3; clean='October' vs corrupted='April'

### Example 4

- **CLEAN**:     `The number after three is`
- **CORRUPTED**: `The number after four is`
- **Difference**: 1/5 tokens differ at positions 3; clean='three' vs corrupted='four'

### Example 5

- **CLEAN**:     `The day after Monday is`
- **CORRUPTED**: `The day after Saturday is`
- **Difference**: 1/5 tokens differ at positions 3; clean='Monday' vs corrupted='Saturday'

### Example 6

- **CLEAN**:     `The month after September is`
- **CORRUPTED**: `The month after November is`
- **Difference**: 1/5 tokens differ at positions 3; clean='September' vs corrupted='November'

### Example 7

- **CLEAN**:     `The day after Wednesday is`
- **CORRUPTED**: `The day after Monday is`
- **Difference**: 1/5 tokens differ at positions 3; clean='Wednesday' vs corrupted='Monday'

### Example 8

- **CLEAN**:     `The number after two is`
- **CORRUPTED**: `The number after eight is`
- **Difference**: 1/5 tokens differ at positions 3; clean='two' vs corrupted='eight'

### Example 9

- **CLEAN**:     `The number after six is`
- **CORRUPTED**: `The number after four is`
- **Difference**: 1/5 tokens differ at positions 3; clean='six' vs corrupted='four'

### Example 10

- **CLEAN**:     `The number after two is`
- **CORRUPTED**: `The number after four is`
- **Difference**: 1/5 tokens differ at positions 3; clean='two' vs corrupted='four'

### Example 11

- **CLEAN**:     `The month after June is`
- **CORRUPTED**: `The month after November is`
- **Difference**: 1/5 tokens differ at positions 3; clean='June' vs corrupted='November'

### Example 12

- **CLEAN**:     `The number after four is`
- **CORRUPTED**: `The number after six is`
- **Difference**: 1/5 tokens differ at positions 3; clean='four' vs corrupted='six'

### Example 13

- **CLEAN**:     `The month after August is`
- **CORRUPTED**: `The month after October is`
- **Difference**: 1/5 tokens differ at positions 3; clean='August' vs corrupted='October'

### Example 14

- **CLEAN**:     `The month after January is`
- **CORRUPTED**: `The month after October is`
- **Difference**: 1/5 tokens differ at positions 3; clean='January' vs corrupted='October'

### Example 15

- **CLEAN**:     `The day after Monday is`
- **CORRUPTED**: `The day after Friday is`
- **Difference**: 1/5 tokens differ at positions 3; clean='Monday' vs corrupted='Friday'

### Example 16

- **CLEAN**:     `The number after one is`
- **CORRUPTED**: `The number after six is`
- **Difference**: 1/5 tokens differ at positions 3; clean='one' vs corrupted='six'

### Example 17

- **CLEAN**:     `The month after June is`
- **CORRUPTED**: `The month after April is`
- **Difference**: 1/5 tokens differ at positions 3; clean='June' vs corrupted='April'

### Example 18

- **CLEAN**:     `The number after six is`
- **CORRUPTED**: `The number after seven is`
- **Difference**: 1/5 tokens differ at positions 3; clean='six' vs corrupted='seven'

### Example 19

- **CLEAN**:     `The day after Tuesday is`
- **CORRUPTED**: `The day after Saturday is`
- **Difference**: 1/5 tokens differ at positions 3; clean='Tuesday' vs corrupted='Saturday'
