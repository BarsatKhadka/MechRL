# Task: greaterthan

- **Model**: gpt2 (n_layers=12, n_heads=12)
- **Batch size**: 20
- **Sequence length**: 12
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: -0.8517
- **Metric aggregate (corrupted)**: 0.2810

Metadata:
```json
{
  "source": "Hanna et al. 2023 (helpers from paperCodes/gpt2-greater-than)",
  "prompt_example": "The siege lasted from the year 1692 to the year 16",
  "corrupted_example": "The siege lasted from the year 1601 to the year 16",
  "full_years": [
    1692,
    1578,
    1365,
    1405,
    1435,
    1226,
    1627,
    1757,
    1164,
    1233,
    1148,
    1408,
    1634,
    1293,
    1806,
    1816,
    1208,
    1346,
    1841,
    1551
  ],
  "target_token_length": 12
}
```

## Examples

Showing first 20 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     `The siege lasted from the year 1692 to the year 16`
- **CORRUPTED**: `The siege lasted from the year 1601 to the year 16`
- **Difference**: 1/12 tokens differ at positions 7; clean='92' vs corrupted='01'
- **Year threshold (YY)**: `92` — model should put probability on YY > 92

### Example 1

- **CLEAN**:     `The reforms lasted from the year 1578 to the year 15`
- **CORRUPTED**: `The reforms lasted from the year 1501 to the year 15`
- **Difference**: 1/12 tokens differ at positions 7; clean='78' vs corrupted='01'
- **Year threshold (YY)**: `78` — model should put probability on YY > 78

### Example 2

- **CLEAN**:     `The fall lasted from the year 1365 to the year 13`
- **CORRUPTED**: `The fall lasted from the year 1301 to the year 13`
- **Difference**: 1/12 tokens differ at positions 7; clean='65' vs corrupted='01'
- **Year threshold (YY)**: `65` — model should put probability on YY > 65

### Example 3

- **CLEAN**:     `The demonstrations lasted from the year 1405 to the year 14`
- **CORRUPTED**: `The demonstrations lasted from the year 1401 to the year 14`
- **Difference**: 1/12 tokens differ at positions 7; clean='05' vs corrupted='01'
- **Year threshold (YY)**: `5` — model should put probability on YY > 05

### Example 4

- **CLEAN**:     `The incarceration lasted from the year 1435 to the year 14`
- **CORRUPTED**: `The incarceration lasted from the year 1401 to the year 14`
- **Difference**: 1/12 tokens differ at positions 7; clean='35' vs corrupted='01'
- **Year threshold (YY)**: `35` — model should put probability on YY > 35

### Example 5

- **CLEAN**:     `The expedition lasted from the year 1226 to the year 12`
- **CORRUPTED**: `The expedition lasted from the year 1201 to the year 12`
- **Difference**: 1/12 tokens differ at positions 7; clean='26' vs corrupted='01'
- **Year threshold (YY)**: `26` — model should put probability on YY > 26

### Example 6

- **CLEAN**:     `The riot lasted from the year 1627 to the year 16`
- **CORRUPTED**: `The riot lasted from the year 1601 to the year 16`
- **Difference**: 1/12 tokens differ at positions 7; clean='27' vs corrupted='01'
- **Year threshold (YY)**: `27` — model should put probability on YY > 27

### Example 7

- **CLEAN**:     `The domination lasted from the year 1757 to the year 17`
- **CORRUPTED**: `The domination lasted from the year 1701 to the year 17`
- **Difference**: 1/12 tokens differ at positions 7; clean='57' vs corrupted='01'
- **Year threshold (YY)**: `57` — model should put probability on YY > 57

### Example 8

- **CLEAN**:     `The illness lasted from the year 1164 to the year 11`
- **CORRUPTED**: `The illness lasted from the year 1101 to the year 11`
- **Difference**: 1/12 tokens differ at positions 7; clean='64' vs corrupted='01'
- **Year threshold (YY)**: `64` — model should put probability on YY > 64

### Example 9

- **CLEAN**:     `The negotiation lasted from the year 1233 to the year 12`
- **CORRUPTED**: `The negotiation lasted from the year 1201 to the year 12`
- **Difference**: 1/12 tokens differ at positions 7; clean='33' vs corrupted='01'
- **Year threshold (YY)**: `33` — model should put probability on YY > 33

### Example 10

- **CLEAN**:     `The testing lasted from the year 1148 to the year 11`
- **CORRUPTED**: `The testing lasted from the year 1101 to the year 11`
- **Difference**: 1/12 tokens differ at positions 7; clean='48' vs corrupted='01'
- **Year threshold (YY)**: `48` — model should put probability on YY > 48

### Example 11

- **CLEAN**:     `The improvement lasted from the year 1408 to the year 14`
- **CORRUPTED**: `The improvement lasted from the year 1401 to the year 14`
- **Difference**: 1/12 tokens differ at positions 7; clean='08' vs corrupted='01'
- **Year threshold (YY)**: `8` — model should put probability on YY > 08

### Example 12

- **CLEAN**:     `The disagreement lasted from the year 1634 to the year 16`
- **CORRUPTED**: `The disagreement lasted from the year 1601 to the year 16`
- **Difference**: 1/12 tokens differ at positions 7; clean='34' vs corrupted='01'
- **Year threshold (YY)**: `34` — model should put probability on YY > 34

### Example 13

- **CLEAN**:     `The reforms lasted from the year 1293 to the year 12`
- **CORRUPTED**: `The reforms lasted from the year 1201 to the year 12`
- **Difference**: 1/12 tokens differ at positions 7; clean='93' vs corrupted='01'
- **Year threshold (YY)**: `93` — model should put probability on YY > 93

### Example 14

- **CLEAN**:     `The order lasted from the year 1806 to the year 18`
- **CORRUPTED**: `The order lasted from the year 1801 to the year 18`
- **Difference**: 1/12 tokens differ at positions 7; clean='06' vs corrupted='01'
- **Year threshold (YY)**: `6` — model should put probability on YY > 06

### Example 15

- **CLEAN**:     `The decrease lasted from the year 1816 to the year 18`
- **CORRUPTED**: `The decrease lasted from the year 1801 to the year 18`
- **Difference**: 1/12 tokens differ at positions 7; clean='16' vs corrupted='01'
- **Year threshold (YY)**: `16` — model should put probability on YY > 16

### Example 16

- **CLEAN**:     `The tests lasted from the year 1208 to the year 12`
- **CORRUPTED**: `The tests lasted from the year 1201 to the year 12`
- **Difference**: 1/12 tokens differ at positions 7; clean='08' vs corrupted='01'
- **Year threshold (YY)**: `8` — model should put probability on YY > 08

### Example 17

- **CLEAN**:     `The voyage lasted from the year 1346 to the year 13`
- **CORRUPTED**: `The voyage lasted from the year 1301 to the year 13`
- **Difference**: 1/12 tokens differ at positions 7; clean='46' vs corrupted='01'
- **Year threshold (YY)**: `46` — model should put probability on YY > 46

### Example 18

- **CLEAN**:     `The romance lasted from the year 1841 to the year 18`
- **CORRUPTED**: `The romance lasted from the year 1801 to the year 18`
- **Difference**: 1/12 tokens differ at positions 7; clean='41' vs corrupted='01'
- **Year threshold (YY)**: `41` — model should put probability on YY > 41

### Example 19

- **CLEAN**:     `The testing lasted from the year 1551 to the year 15`
- **CORRUPTED**: `The testing lasted from the year 1501 to the year 15`
- **Difference**: 1/12 tokens differ at positions 7; clean='51' vs corrupted='01'
- **Year threshold (YY)**: `51` — model should put probability on YY > 51
