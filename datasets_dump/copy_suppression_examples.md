# Task: copy_suppression

- **Model**: gpt2 (n_layers=12, n_heads=12)
- **Batch size**: 20
- **Sequence length**: 18
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: 18.5172
- **Metric aggregate (corrupted)**: 12.0656

Metadata:
```json
{
  "source": "McDougall et al. 2023 (synthetic templated version)",
  "canonical_head": "L10.H7",
  "task": "measure suppression of recently-mentioned token",
  "n_names": 20,
  "n_foods": 8
}
```

## Examples

Showing first 20 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     `<|endoftext|>Henry and Sophia are friends. Henry was eating a sandwich. Sophia was also eating a`
- **CORRUPTED**: `<|endoftext|>Henry and Sophia are friends. Henry was eating a burger. Sophia was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='sandwich' vs corrupted='burger'

### Example 1

- **CLEAN**:     `<|endoftext|>Charles and Emma are friends. Charles was eating a taco. Emma was also eating a`
- **CORRUPTED**: `<|endoftext|>Charles and Emma are friends. Charles was eating a wrap. Emma was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='taco' vs corrupted='wrap'

### Example 2

- **CLEAN**:     `<|endoftext|>Laura and Emma are friends. Laura was eating a steak. Emma was also eating a`
- **CORRUPTED**: `<|endoftext|>Laura and Emma are friends. Laura was eating a salad. Emma was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='steak' vs corrupted='salad'

### Example 3

- **CLEAN**:     `<|endoftext|>David and Charles are friends. David was eating a burger. Charles was also eating a`
- **CORRUPTED**: `<|endoftext|>David and Charles are friends. David was eating a wrap. Charles was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='burger' vs corrupted='wrap'

### Example 4

- **CLEAN**:     `<|endoftext|>Michael and Sarah are friends. Michael was eating a salad. Sarah was also eating a`
- **CORRUPTED**: `<|endoftext|>Michael and Sarah are friends. Michael was eating a wrap. Sarah was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='salad' vs corrupted='wrap'

### Example 5

- **CLEAN**:     `Grace and Michael are friends. Grace was eating a salad. Michael was also eating a`
- **CORRUPTED**: `Grace and Michael are friends. Grace was eating a sandwich. Michael was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='salad' vs corrupted='sandwich'

### Example 6

- **CLEAN**:     `<|endoftext|>James and Daniel are friends. James was eating a wrap. Daniel was also eating a`
- **CORRUPTED**: `<|endoftext|>James and Daniel are friends. James was eating a salad. Daniel was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='wrap' vs corrupted='salad'

### Example 7

- **CLEAN**:     `<|endoftext|>Sarah and Olivia are friends. Sarah was eating a taco. Olivia was also eating a`
- **CORRUPTED**: `<|endoftext|>Sarah and Olivia are friends. Sarah was eating a burger. Olivia was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='taco' vs corrupted='burger'

### Example 8

- **CLEAN**:     `Grace and David are friends. Grace was eating a wrap. David was also eating a`
- **CORRUPTED**: `Grace and David are friends. Grace was eating a cookie. David was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='wrap' vs corrupted='cookie'

### Example 9

- **CLEAN**:     `<|endoftext|>Charles and Robert are friends. Charles was eating a sandwich. Robert was also eating a`
- **CORRUPTED**: `<|endoftext|>Charles and Robert are friends. Charles was eating a taco. Robert was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='sandwich' vs corrupted='taco'

### Example 10

- **CLEAN**:     `<|endoftext|>Alice and John are friends. Alice was eating a pizza. John was also eating a`
- **CORRUPTED**: `<|endoftext|>Alice and John are friends. Alice was eating a steak. John was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='pizza' vs corrupted='steak'

### Example 11

- **CLEAN**:     `<|endoftext|>Henry and John are friends. Henry was eating a wrap. John was also eating a`
- **CORRUPTED**: `<|endoftext|>Henry and John are friends. Henry was eating a taco. John was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='wrap' vs corrupted='taco'

### Example 12

- **CLEAN**:     `<|endoftext|>Daniel and Anna are friends. Daniel was eating a steak. Anna was also eating a`
- **CORRUPTED**: `<|endoftext|>Daniel and Anna are friends. Daniel was eating a wrap. Anna was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='steak' vs corrupted='wrap'

### Example 13

- **CLEAN**:     `<|endoftext|>James and David are friends. James was eating a cookie. David was also eating a`
- **CORRUPTED**: `<|endoftext|>James and David are friends. James was eating a pizza. David was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='cookie' vs corrupted='pizza'

### Example 14

- **CLEAN**:     `<|endoftext|>Michael and Alice are friends. Michael was eating a wrap. Alice was also eating a`
- **CORRUPTED**: `<|endoftext|>Michael and Alice are friends. Michael was eating a sandwich. Alice was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='wrap' vs corrupted='sandwich'

### Example 15

- **CLEAN**:     `<|endoftext|>James and Daniel are friends. James was eating a wrap. Daniel was also eating a`
- **CORRUPTED**: `<|endoftext|>James and Daniel are friends. James was eating a sandwich. Daniel was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='wrap' vs corrupted='sandwich'

### Example 16

- **CLEAN**:     `<|endoftext|>Laura and Alice are friends. Laura was eating a salad. Alice was also eating a`
- **CORRUPTED**: `<|endoftext|>Laura and Alice are friends. Laura was eating a steak. Alice was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='salad' vs corrupted='steak'

### Example 17

- **CLEAN**:     `<|endoftext|>Sarah and Alice are friends. Sarah was eating a steak. Alice was also eating a`
- **CORRUPTED**: `<|endoftext|>Sarah and Alice are friends. Sarah was eating a taco. Alice was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='steak' vs corrupted='taco'

### Example 18

- **CLEAN**:     `<|endoftext|>Alice and David are friends. Alice was eating a salad. David was also eating a`
- **CORRUPTED**: `<|endoftext|>Alice and David are friends. Alice was eating a cookie. David was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='salad' vs corrupted='cookie'

### Example 19

- **CLEAN**:     `<|endoftext|>James and Henry are friends. James was eating a steak. Henry was also eating a`
- **CORRUPTED**: `<|endoftext|>James and Henry are friends. James was eating a salad. Henry was also eating a`
- **Difference**: 1/18 tokens differ at positions 11; clean='steak' vs corrupted='salad'
