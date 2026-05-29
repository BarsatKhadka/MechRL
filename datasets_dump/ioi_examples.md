# Task: ioi

- **Model**: gpt2 (n_layers=12, n_heads=12)
- **Batch size**: 20
- **Sequence length**: 15
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: -3.5696
- **Metric aggregate (corrupted)**: -0.2853

Metadata:
```json
{
  "source": "acdc.ioi.utils.get_all_ioi_things",
  "prompt_type": "ABBA"
}
```

## Examples

Showing first 20 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     `Then, Mary and Jeffrey went to the restaurant. Jeffrey gave a bone to`
- **CORRUPTED**: `Then, Melissa and Nicole went to the restaurant. Anthony gave a bone to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Mary'|'Jeffrey'|'Jeffrey' vs corrupted='Melissa'|'Nicole'|'Anthony'
- **Correct answer token**: `5335` -> `' Mary'`

### Example 1

- **CLEAN**:     `Then, Brittany and Jamie went to the garden. Jamie gave a necklace to`
- **CORRUPTED**: `Then, Lauren and Alicia went to the garden. Daniel gave a necklace to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Brittany'|'Jamie'|'Jamie' vs corrupted='Lauren'|'Alicia'|'Daniel'
- **Correct answer token**: `48773` -> `' Brittany'`

### Example 2

- **CLEAN**:     `Then, Christine and Amy went to the station. Amy gave a drink to`
- **CORRUPTED**: `Then, Thomas and Elizabeth went to the station. Jesse gave a drink to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Christine'|'Amy'|'Amy' vs corrupted='Thomas'|'Elizabeth'|'Jesse'
- **Correct answer token**: `26088` -> `' Christine'`

### Example 3

- **CLEAN**:     `Then, Jonathan and Adam went to the garden. Adam gave a bone to`
- **CORRUPTED**: `Then, Kelly and Nathan went to the garden. Travis gave a bone to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Jonathan'|'Adam'|'Adam' vs corrupted='Kelly'|'Nathan'|'Travis'
- **Correct answer token**: `11232` -> `' Jonathan'`

### Example 4

- **CLEAN**:     `Then, Katie and Crystal went to the garden. Crystal gave a computer to`
- **CORRUPTED**: `Then, Brittany and Christina went to the garden. Jeremy gave a computer to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Katie'|'Crystal'|'Crystal' vs corrupted='Brittany'|'Christina'|'Jeremy'
- **Correct answer token**: `24721` -> `' Katie'`

### Example 5

- **CLEAN**:     `Then, Alexander and Eric went to the house. Eric gave a kiss to`
- **CORRUPTED**: `Then, Lindsey and Stephanie went to the house. Jamie gave a kiss to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Alexander'|'Eric'|'Eric' vs corrupted='Lindsey'|'Stephanie'|'Jamie'
- **Correct answer token**: `10009` -> `' Alexander'`

### Example 6

- **CLEAN**:     `Then, Anthony and Michelle went to the office. Michelle gave a ring to`
- **CORRUPTED**: `Then, Samantha and Angela went to the office. Matthew gave a ring to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Anthony'|'Michelle'|'Michelle' vs corrupted='Samantha'|'Angela'|'Matthew'
- **Correct answer token**: `9953` -> `' Anthony'`

### Example 7

- **CLEAN**:     `Then, Shannon and Michael went to the store. Michael gave a drink to`
- **CORRUPTED**: `Then, Jesse and Erica went to the store. Mark gave a drink to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Shannon'|'Michael'|'Michael' vs corrupted='Jesse'|'Erica'|'Mark'
- **Correct answer token**: `28108` -> `' Shannon'`

### Example 8

- **CLEAN**:     `Then, Jason and Matthew went to the house. Matthew gave a kiss to`
- **CORRUPTED**: `Then, Cody and Angela went to the house. David gave a kiss to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Jason'|'Matthew'|'Matthew' vs corrupted='Cody'|'Angela'|'David'
- **Correct answer token**: `8982` -> `' Jason'`

### Example 9

- **CLEAN**:     `Then, Ryan and Bradley went to the school. Bradley gave a basketball to`
- **CORRUPTED**: `Then, Eric and Patrick went to the school. Lindsey gave a basketball to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Ryan'|'Bradley'|'Bradley' vs corrupted='Eric'|'Patrick'|'Lindsey'
- **Correct answer token**: `6047` -> `' Ryan'`

### Example 10

- **CLEAN**:     `Then, Christopher and James went to the store. James gave a kiss to`
- **CORRUPTED**: `Then, Richard and Daniel went to the store. Kimberly gave a kiss to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Christopher'|'James'|'James' vs corrupted='Richard'|'Daniel'|'Kimberly'
- **Correct answer token**: `12803` -> `' Christopher'`

### Example 11

- **CLEAN**:     `Then, Sara and Nicholas went to the garden. Nicholas gave a drink to`
- **CORRUPTED**: `Then, Michael and Allison went to the garden. Laura gave a drink to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Sara'|'Nicholas'|'Nicholas' vs corrupted='Michael'|'Allison'|'Laura'
- **Correct answer token**: `24799` -> `' Sara'`

### Example 12

- **CLEAN**:     `Then, Kenneth and Bradley went to the restaurant. Bradley gave a bone to`
- **CORRUPTED**: `Then, Jacob and John went to the restaurant. Allison gave a bone to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Kenneth'|'Bradley'|'Bradley' vs corrupted='Jacob'|'John'|'Allison'
- **Correct answer token**: `23632` -> `' Kenneth'`

### Example 13

- **CLEAN**:     `Then, Joshua and Steven went to the restaurant. Steven gave a computer to`
- **CORRUPTED**: `Then, Benjamin and Timothy went to the restaurant. Richard gave a computer to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Joshua'|'Steven'|'Steven' vs corrupted='Benjamin'|'Timothy'|'Richard'
- **Correct answer token**: `20700` -> `' Joshua'`

### Example 14

- **CLEAN**:     `Then, Ryan and Shannon went to the school. Shannon gave a bone to`
- **CORRUPTED**: `Then, Brittany and Adam went to the school. Mark gave a bone to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Ryan'|'Shannon'|'Shannon' vs corrupted='Brittany'|'Adam'|'Mark'
- **Correct answer token**: `6047` -> `' Ryan'`

### Example 15

- **CLEAN**:     `Then, Alicia and Kimberly went to the garden. Kimberly gave a necklace to`
- **CORRUPTED**: `Then, Jesse and Erin went to the garden. Brandon gave a necklace to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Alicia'|'Kimberly'|'Kimberly' vs corrupted='Jesse'|'Erin'|'Brandon'
- **Correct answer token**: `39607` -> `' Alicia'`

### Example 16

- **CLEAN**:     `Then, Katherine and Kimberly went to the school. Kimberly gave a drink to`
- **CORRUPTED**: `Then, Alexander and Samantha went to the school. Samantha gave a drink to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Katherine'|'Kimberly'|'Kimberly' vs corrupted='Alexander'|'Samantha'|'Samantha'
- **Correct answer token**: `32719` -> `' Katherine'`

### Example 17

- **CLEAN**:     `Then, Scott and Cody went to the house. Cody gave a necklace to`
- **CORRUPTED**: `Then, Benjamin and Jason went to the house. Emily gave a necklace to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Scott'|'Cody'|'Cody' vs corrupted='Benjamin'|'Jason'|'Emily'
- **Correct answer token**: `4746` -> `' Scott'`

### Example 18

- **CLEAN**:     `Then, Emily and Andrew went to the school. Andrew gave a snack to`
- **CORRUPTED**: `Then, Samantha and Ashley went to the school. Amber gave a snack to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Emily'|'Andrew'|'Andrew' vs corrupted='Samantha'|'Ashley'|'Amber'
- **Correct answer token**: `17608` -> `' Emily'`

### Example 19

- **CLEAN**:     `Then, Lauren and Robert went to the hospital. Robert gave a kiss to`
- **CORRUPTED**: `Then, Katherine and Jason went to the hospital. Richard gave a kiss to`
- **Difference**: 3/15 tokens differ at positions 2, 4, 10; clean='Lauren'|'Robert'|'Robert' vs corrupted='Katherine'|'Jason'|'Richard'
- **Correct answer token**: `25672` -> `' Lauren'`
