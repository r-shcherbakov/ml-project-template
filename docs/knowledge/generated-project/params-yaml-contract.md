# Pipeline Step Parameters Contract

## Context

Each pipeline step that has configurable parameters defines a `XxxParams(BaseModel)`
class in the **same file** as the step class. Parameters are passed as a **required**
constructor argument — there is no default. `BasePipelineStep._connect_params()` calls
`task.connect(params.model_dump())` to make parameters visible and editable in the
ClearML UI without code changes.

## Invariants

1. `XxxParams` inherits from `pydantic.BaseModel` and lives in the same file as its step.
2. `params` is a **required** constructor argument — `XxxStep()` with no args raises `TypeError`.
3. `pipeline.py` always instantiates params explicitly (never relying on defaults alone).
4. `self.step_params` is always a typed `XxxParams` instance (or `None` for param-free steps).
5. `task.connect()` receives `self.step_params.model_dump()` — ClearML UI override behaviour is unchanged.
6. Never mutate `self.step_params` after init — use instance variables for runtime-computed values.

## How It Works

```python
# src/preprocess/preprocess_pipeline_step.py
class PreprocessParams(BaseModel):
    skip_mark: bool = False          # typed, IDE-complete, validated

class PreprocessPipelineStep(BasePipelineStep):
    def __init__(self, params: PreprocessParams):
        super().__init__(PREPROCESS, params=params)

    def start(self, dataset_id: str) -> str:
        if self.step_params.skip_mark:   # attribute access, not dict .get()
            ...
```

```python
# src/pipelines/pipeline.py
def run_preprocess_step(dataset_id: str) -> str:
    return PreprocessPipelineStep(
        params=PreprocessParams(skip_mark=False),
    ).start(dataset_id=dataset_id)
```

## Steps without parameters

Steps that need no parameters (e.g. `FeatureEngineerPipelineStep`) simply omit `params`:

```python
class FeatureEngineerPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(FEATURE_ENGINEER)   # params defaults to None
```

`_connect_params()` is a no-op when `self.step_params is None`.

## Agent Checklist

Before adding or modifying parameters:
- [ ] Is `XxxParams(BaseModel)` defined in the same file as the step?
- [ ] Is `params` a required constructor argument (no default)?
- [ ] Does `pipeline.py` pass params explicitly at the call site?
- [ ] Does step code use `self.step_params.field_name` (not `.get()`)?
- [ ] Are runtime-computed values stored as instance vars (not mutating `self.step_params`)?
