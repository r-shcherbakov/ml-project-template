# Design: Explicit Pipeline Step Parameters

**Date:** 2026-04-23  
**Status:** Approved

## Problem

`params.yaml` introduces four compounding issues:

1. `self.step_params` is `Optional[dict]` — no types, no IDE completion, `.get()` returns `Any`
2. Typos in YAML keys fail silently at runtime (missing key → `None`, not an error)
3. Step parameters are invisible without opening a separate file
4. Defaults are split between YAML values and `.get("key", fallback)` calls in step code

## Decision

Replace `params.yaml` with typed Pydantic `BaseModel` classes defined alongside each step. Parameters are injected via the step constructor as a required argument — calling a step without params is a `TypeError`.

## Design

### `BasePipelineStep` changes

`_init_parameters()` is removed. A new `_connect_params()` method handles ClearML
registration using `params.model_dump()`:

```python
class BasePipelineStep(ABC):
    def __init__(
        self,
        pipeline_step: 'PipelineStep',
        params: Optional[BaseModel] = None,
    ):
        self.pipeline_step = pipeline_step
        self.settings = SETTINGS
        self.step_params = params
        self._init_task()
        self._connect_params()

    def _connect_params(self):
        if self.step_params is not None:
            self.task.connect(
                self.step_params.model_dump(),
                name=self.pipeline_step.name.replace('_', ' '),
            )
```

- `yaml` import and `self.settings.params_path` usage are removed from the base class.
- Steps without parameters pass nothing; `_connect_params()` is a no-op.

### Per-step `Params` model

Each step that has parameters defines a `Params` Pydantic model in the **same file** as
the step class. The step constructor takes `params` as a **required argument** (no default):

```python
# src/preprocess/preprocess_step.py
from pydantic import BaseModel

class PreprocessParams(BaseModel):
    skip_mark: bool = False

class PreprocessPipelineStep(BasePipelineStep):
    def __init__(self, params: PreprocessParams):
        super().__init__(PREPROCESS, params=params)

    def start(self, dataset_id: str) -> str:
        if self.step_params.skip_mark:   # typed, IDE-complete
            ...
```

Steps without parameters omit `params` entirely and call `super().__init__(STEP)`.

### `pipeline.py` — always explicit

Every step instantiation in `pipeline.py` passes params explicitly:

```python
def run_preprocess_step(dataset_id: str) -> str:
    return PreprocessPipelineStep(
        params=PreprocessParams(skip_mark=False),
    ).start(dataset_id=dataset_id)
```

This makes `pipeline.py` the single place to read the full parameter configuration of
a pipeline run, without opening any other file.

### `common` parameters — removed

The `common` YAML section (`mandatory_features`, `forbidden_features`) is removed.
These values already live in `src/common/features.py` as `MANDATORY_FEATURES` etc.
Step code that needs them imports directly from there.

### Files removed / changed

| File | Action |
|------|--------|
| `src/params.yaml` | **deleted** |
| `src/settings.py` | remove `params_path` field |
| `src/core/pipeline_step.py` | replace `_init_parameters()` with `_connect_params()`, update `__init__` signature |
| `src/pipelines/pipeline.py` | pass explicit `Params(...)` at every step call site |
| Each `*_step.py` | add `*Params(BaseModel)`, update `__init__`, replace `self.step_params.get(...)` with attribute access |

## Invariants

1. Every step with parameters defines `XxxParams(BaseModel)` in its own file.
2. `params` is a required constructor argument — no optional default allowed.
3. `pipeline.py` always instantiates params explicitly, never relying on model defaults alone.
4. `self.step_params` is always a typed model instance (or `None` for param-free steps) — never a raw dict.
5. `task.connect()` receives `self.step_params.model_dump()` — ClearML UI override behaviour is unchanged.

## What does NOT change

- ClearML parameter visibility and override in the UI — `task.connect(dict)` still works identically.
- `BasePipelineStep` contract for step subclasses (`start()` signature, dataset up/download, logging).
- `PipelineStep` dataclass and `pipeline_steps.py`.
