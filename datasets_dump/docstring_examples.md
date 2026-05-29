# Task: docstring

- **Model**: Attn_Only_4L512W_C4_Code (n_layers=4, n_heads=8)
- **Batch size**: 10
- **Sequence length**: 41
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: 0.0000
- **Metric aggregate (corrupted)**: 4.1946

Metadata:
```json
{
  "source": "acdc.docstring.utils.get_all_docstring_things",
  "model": "attn-only-4l (NOT GPT-2 small)"
}
```

## Examples

Showing first 10 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     `<|BOS|>
def old(self, first, files, page, names, size, read):
    """sector gap population

    :param page: message tree
    :param names: detail mine
    :param`
- **CORRUPTED**: `<|BOS|>
def old(self, first, files, project, target, new, read):
    """sector gap population

    :param image: message tree
    :param update: detail mine
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='page'|'names'|'size'|'page'|'names' vs corrupted='project'|'target'|'new'|'image'|'update'

### Example 1

- **CLEAN**:     `<|BOS|>
def model(self, fields, page, table, config, count, number):
    """unit bone paper

    :param table: selection sky
    :param config: host action
    :param`
- **CORRUPTED**: `<|BOS|>
def model(self, fields, page, old, option, project, number):
    """unit bone paper

    :param user: selection sky
    :param context: host action
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='table'|'config'|'count'|'table'|'config' vs corrupted='old'|'option'|'project'|'user'|'context'

### Example 2

- **CLEAN**:     `<|BOS|>
def root(self, event, context, task, tag, resource, base):
    """stone delay sense

    :param task: reply sea
    :param tag: search wire
    :param`
- **CORRUPTED**: `<|BOS|>
def root(self, event, context, line, request, default, base):
    """stone delay sense

    :param form: reply sea
    :param names: search wire
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='task'|'tag'|'resource'|'task'|'tag' vs corrupted='line'|'request'|'default'|'form'|'names'

### Example 3

- **CLEAN**:     `<|BOS|>
def error(self, image, client, user, name, port, index):
    """dog request wheel

    :param user: travel creation
    :param name: notice bell
    :param`
- **CORRUPTED**: `<|BOS|>
def error(self, image, client, content, url, query, index):
    """dog request wheel

    :param base: travel creation
    :param config: notice bell
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='user'|'name'|'port'|'user'|'name' vs corrupted='content'|'url'|'query'|'base'|'config'

### Example 4

- **CLEAN**:     `<|BOS|>
def error(self, create, option, file, run, client, project):
    """land employment camp

    :param file: protein author
    :param run: forest degree
    :param`
- **CORRUPTED**: `<|BOS|>
def error(self, create, option, output, host, label, project):
    """land employment camp

    :param first: protein author
    :param text: forest degree
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='file'|'run'|'client'|'file'|'run' vs corrupted='output'|'host'|'label'|'first'|'text'

### Example 5

- **CLEAN**:     `<|BOS|>
def color(self, current, result, value, msg, valid, version):
    """share football pub

    :param value: mouth sheet
    :param msg: export chapter
    :param`
- **CORRUPTED**: `<|BOS|>
def color(self, current, result, description, files, header, version):
    """share football pub

    :param call: mouth sheet
    :param port: export chapter
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='value'|'msg'|'valid'|'value'|'msg' vs corrupted='description'|'files'|'header'|'call'|'port'

### Example 6

- **CLEAN**:     `<|BOS|>
def header(self, config, description, response, match, command, query):
    """master lot network

    :param response: cancer element
    :param match: wave saving
    :param`
- **CORRUPTED**: `<|BOS|>
def header(self, config, description, load, default, required, query):
    """master lot network

    :param project: cancer element
    :param image: wave saving
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='response'|'match'|'command'|'response'|'match' vs corrupted='load'|'default'|'required'|'project'|'image'

### Example 7

- **CLEAN**:     `<|BOS|>
def row(self, ret, values, form, read, line, test):
    """tax sum lead

    :param form: parent talk
    :param read: nature camera
    :param`
- **CORRUPTED**: `<|BOS|>
def row(self, ret, values, label, count, result, test):
    """tax sum lead

    :param key: parent talk
    :param load: nature camera
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='form'|'read'|'line'|'form'|'read' vs corrupted='label'|'count'|'result'|'key'|'load'

### Example 8

- **CLEAN**:     `<|BOS|>
def index(self, number, run, current, task, ret, found):
    """century border bone

    :param current: flow selection
    :param task: round actor
    :param`
- **CORRUPTED**: `<|BOS|>
def index(self, number, run, fields, url, field, found):
    """century border bone

    :param token: flow selection
    :param new: round actor
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='current'|'task'|'ret'|'current'|'task' vs corrupted='fields'|'url'|'field'|'token'|'new'

### Example 9

- **CLEAN**:     `<|BOS|>
def msg(self, default, obj, parser, ret, base, client):
    """dark language session

    :param parser: technology castle
    :param ret: guide statement
    :param`
- **CORRUPTED**: `<|BOS|>
def msg(self, default, obj, query, message, model, client):
    """dark language session

    :param string: technology castle
    :param root: guide statement
    :param`
- **Difference**: 5/41 tokens differ at positions 11, 13, 15, 27, 34; clean='parser'|'ret'|'base'|'parser'|'ret' vs corrupted='query'|'message'|'model'|'string'|'root'
