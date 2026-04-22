# params.yaml Contract

## Context

`src/params.yaml` is the single file for all pipeline step parameters.
`BasePipelineStep._init_parameters()` reads it automatically on step
initialization and connects the matching section to the ClearML task —
making parameters visible and editable in the ClearML UI without code changes.
Parameters can then be overridden in ClearML UI for remote runs.

## Invariants

1. The top-level key in `params.yaml` must **exactly** match `PipelineStep.name` (case-sensitive, underscores preserved).
2. `common` is a reserved top-level key — its contents are connected to every task as the "common" parameter group. Use it only for truly cross-step parameters.
3. A missing step key results in `self.step_params = None` — no error is raised, but the step has no ClearML parameter binding.
4. All parameter values must be YAML-serializable primitives: `str`, `int`, `float`, `bool`, `list`, `dict`.
5. Never read `params.yaml` directly in step code — always use `self.step_params` (already loaded in `_init_parameters()`).

## How It Works

```yaml
# src/params.yaml structure
common:              # ← connected to ALL tasks as "common" parameter group
  mandatory_features:
    - target
  forbidden_features:
    - GROUP_ID

preprocess:          # ← PipelineStep.name = "preprocess" → self.step_params
  skip_mark: False   #    access: self.step_params.get("skip_mark", True)

feature_engineer:    # ← PipelineStep.name = "feature_engineer"
  some_int_parameter: 1

train:
  skip_cv: True
  n_splits: 4
  train_final_model: True
```

In `BasePipelineStep._init_parameters()`:
```python
with open(self.settings.params_path) as file:
    params = yaml.load(file, Loader=yaml.Loader)
    self.common_params = params.get('common', None)
    self.step_params = params.get(self.pipeline_step.name, None)  # key = PipelineStep.name

if self.common_params:
    self.task.connect(self.common_params, name="common")
if self.step_params:
    self.task.connect(self.step_params, name=self.pipeline_step.name)
```

## Agent Checklist

Before adding or modifying parameters:
- [ ] Is the top-level key identical to `PipelineStep.name`?
- [ ] Is `common` used only for parameters needed by multiple steps?
- [ ] Are all values YAML-serializable primitives?
- [ ] Is `self.step_params` used in step code (not direct `yaml.load()` calls)?
- [ ] If a step has no parameters, is omitting its key acceptable (results in `self.step_params = None`)?
