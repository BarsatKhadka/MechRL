# Task: docstring_gpt2

- **Model**: gpt2 (n_layers=12, n_heads=12)
- **Batch size**: 20
- **Sequence length**: 41
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: -4.2556
- **Metric aggregate (corrupted)**: 0.3991

Metadata:
```json
{
  "source": "adapted from Heimersheim & Janiak (2023) for GPT-2",
  "task": "predict next docstring argument name",
  "arg_pool_size": 43
}
```

## Examples

Showing first 20 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     `def f(self, level, port, name, path):
    """summary
    :param port:
    :param name:
    :param`
- **CORRUPTED**: `def f(self, level, name, port, limit):
    """summary
    :param name:
    :param port:
    :param`
- **Difference**: 5/41 tokens differ at positions 7, 9, 11, 25, 33; clean='port'|'name'|'path'|'port'|'name' vs corrupted='name'|'port'|'limit'|'name'|'port'

### Example 1

- **CLEAN**:     `def f(self, count, context, output, limit):
    """summary
    :param context:
    :param output:
    :param`
- **CORRUPTED**: `def f(self, shape, limit, count, context):
    """summary
    :param limit:
    :param count:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='count'|'context'|'output'|'limit'|'context' vs corrupted='shape'|'limit'|'count'|'context'|'limit'

### Example 2

- **CLEAN**:     `def f(self, stream, request, index, type):
    """summary
    :param request:
    :param index:
    :param`
- **CORRUPTED**: `def f(self, type, state, request, index):
    """summary
    :param state:
    :param request:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='stream'|'request'|'index'|'type'|'request' vs corrupted='type'|'state'|'request'|'index'|'state'

### Example 3

- **CLEAN**:     `def f(self, count, token, mode, header):
    """summary
    :param token:
    :param mode:
    :param`
- **CORRUPTED**: `def f(self, filter, mode, header, token):
    """summary
    :param mode:
    :param header:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='count'|'token'|'mode'|'header'|'token' vs corrupted='filter'|'mode'|'header'|'token'|'mode'

### Example 4

- **CLEAN**:     `def f(self, version, filter, data, header):
    """summary
    :param filter:
    :param data:
    :param`
- **CORRUPTED**: `def f(self, version, header, query, filter):
    """summary
    :param header:
    :param query:
    :param`
- **Difference**: 5/41 tokens differ at positions 7, 9, 11, 25, 33; clean='filter'|'data'|'header'|'filter'|'data' vs corrupted='header'|'query'|'filter'|'header'|'query'

### Example 5

- **CLEAN**:     `def f(self, key, input, client, label):
    """summary
    :param input:
    :param client:
    :param`
- **CORRUPTED**: `def f(self, client, input, label, args):
    """summary
    :param input:
    :param label:
    :param`
- **Difference**: 4/41 tokens differ at positions 5, 9, 11, 33; clean='key'|'client'|'label'|'client' vs corrupted='client'|'label'|'args'|'label'

### Example 6

- **CLEAN**:     `def f(self, session, table, stream, output):
    """summary
    :param table:
    :param stream:
    :param`
- **CORRUPTED**: `def f(self, output, session, table, stream):
    """summary
    :param session:
    :param table:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='session'|'table'|'stream'|'output'|'table' vs corrupted='output'|'session'|'table'|'stream'|'session'

### Example 7

- **CLEAN**:     `def f(self, request, level, mode, client):
    """summary
    :param level:
    :param mode:
    :param`
- **CORRUPTED**: `def f(self, client, args, request, level):
    """summary
    :param args:
    :param request:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='request'|'level'|'mode'|'client'|'level' vs corrupted='client'|'args'|'request'|'level'|'args'

### Example 8

- **CLEAN**:     `def f(self, name, header, version, path):
    """summary
    :param header:
    :param version:
    :param`
- **CORRUPTED**: `def f(self, header, path, version, field):
    """summary
    :param path:
    :param version:
    :param`
- **Difference**: 4/41 tokens differ at positions 5, 7, 11, 25; clean='name'|'header'|'path'|'header' vs corrupted='header'|'path'|'field'|'path'

### Example 9

- **CLEAN**:     `def f(self, index, name, result, stream):
    """summary
    :param name:
    :param result:
    :param`
- **CORRUPTED**: `def f(self, name, index, stream, result):
    """summary
    :param index:
    :param stream:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='index'|'name'|'result'|'stream'|'name' vs corrupted='name'|'index'|'stream'|'result'|'index'

### Example 10

- **CLEAN**:     `def f(self, output, context, port, node):
    """summary
    :param context:
    :param port:
    :param`
- **CORRUPTED**: `def f(self, image, context, output, port):
    """summary
    :param context:
    :param output:
    :param`
- **Difference**: 4/41 tokens differ at positions 5, 9, 11, 33; clean='output'|'port'|'node'|'port' vs corrupted='image'|'output'|'port'|'output'

### Example 11

- **CLEAN**:     `def f(self, header, size, query, context):
    """summary
    :param size:
    :param query:
    :param`
- **CORRUPTED**: `def f(self, context, filter, header, size):
    """summary
    :param filter:
    :param header:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='header'|'size'|'query'|'context'|'size' vs corrupted='context'|'filter'|'header'|'size'|'filter'

### Example 12

- **CLEAN**:     `def f(self, node, size, label, score):
    """summary
    :param size:
    :param label:
    :param`
- **CORRUPTED**: `def f(self, score, node, size, label):
    """summary
    :param node:
    :param size:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='node'|'size'|'label'|'score'|'size' vs corrupted='score'|'node'|'size'|'label'|'node'

### Example 13

- **CLEAN**:     `def f(self, file, size, filter, input):
    """summary
    :param size:
    :param filter:
    :param`
- **CORRUPTED**: `def f(self, filter, input, size, file):
    """summary
    :param input:
    :param size:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='file'|'size'|'filter'|'input'|'size' vs corrupted='filter'|'input'|'size'|'file'|'input'

### Example 14

- **CLEAN**:     `def f(self, score, size, name, request):
    """summary
    :param size:
    :param name:
    :param`
- **CORRUPTED**: `def f(self, request, score, name, file):
    """summary
    :param score:
    :param name:
    :param`
- **Difference**: 4/41 tokens differ at positions 5, 7, 11, 25; clean='score'|'size'|'request'|'size' vs corrupted='request'|'score'|'file'|'score'

### Example 15

- **CLEAN**:     `def f(self, field, output, value, file):
    """summary
    :param output:
    :param value:
    :param`
- **CORRUPTED**: `def f(self, value, stream, output, field):
    """summary
    :param stream:
    :param output:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='field'|'output'|'value'|'file'|'output' vs corrupted='value'|'stream'|'output'|'field'|'stream'

### Example 16

- **CLEAN**:     `def f(self, label, key, format, type):
    """summary
    :param key:
    :param format:
    :param`
- **CORRUPTED**: `def f(self, format, count, label, key):
    """summary
    :param count:
    :param label:
    :param`
- **Difference**: 6/41 tokens differ at positions 5, 7, 9, 11, 25 (+1 more); clean='label'|'key'|'format'|'type'|'key' vs corrupted='format'|'count'|'label'|'key'|'count'

### Example 17

- **CLEAN**:     `def f(self, name, request, state, host):
    """summary
    :param request:
    :param state:
    :param`
- **CORRUPTED**: `def f(self, name, request, host, input):
    """summary
    :param request:
    :param host:
    :param`
- **Difference**: 3/41 tokens differ at positions 9, 11, 33; clean='state'|'host'|'state' vs corrupted='host'|'input'|'host'

### Example 18

- **CLEAN**:     `def f(self, client, config, output, value):
    """summary
    :param config:
    :param output:
    :param`
- **CORRUPTED**: `def f(self, value, config, client, output):
    """summary
    :param config:
    :param client:
    :param`
- **Difference**: 4/41 tokens differ at positions 5, 9, 11, 33; clean='client'|'output'|'value'|'output' vs corrupted='value'|'client'|'output'|'client'

### Example 19

- **CLEAN**:     `def f(self, version, model, data, field):
    """summary
    :param model:
    :param data:
    :param`
- **CORRUPTED**: `def f(self, version, model, field, data):
    """summary
    :param model:
    :param field:
    :param`
- **Difference**: 3/41 tokens differ at positions 9, 11, 33; clean='data'|'field'|'data' vs corrupted='field'|'data'|'field'
