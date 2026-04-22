# How to Add a Pipeline Step

## Context

Pipeline steps are the fundamental unit of work in generated projects. Each step
downloads its input dataset from ClearML, processes data, and uploads an output
dataset. A step must be declared as a `PipelineStep` constant, implemented as a
`BasePipelineStep` subclass, registered in `pipeline.py`, and have its parameters
in `params.yaml`.

## Invariants

1. `PipelineStep.name` must exactly match the top-level key in `params.yaml` (case-sensitive).
2. `start(dataset_id: str) -> str` — this signature is mandatory. The return value is the output dataset_id.
3. Output directories must be declared as `@computed_field` in `StorageSettings`, not created ad-hoc in step code.
4. A step is optional — omitting it from `pipeline.py` is valid. Including it requires all its `parents` to be included too.

## How It Works: 5-Step Recipe

### Step 1 — Declare the `PipelineStep` constant

In `src/common/pipeline_steps.py`:

```python
from src.settings import StorageSettings
from clearml import TaskTypes

storage_settings = StorageSettings()  # already at top of file

MY_STEP = PipelineStep(
    name="my_step",                               # must match params.yaml key exactly
    task_type=TaskTypes.data_processing.name,     # or training, service, optimizer
    input_directory=storage_settings.processed_folder,
    output_directory=storage_settings.my_step_folder,  # add this field to StorageSettings first
)
```

### Step 2 — Add output directory to `StorageSettings`

In `src/settings.py` (or `settings.py.jinja` in template), inside `StorageSettings`:

```python
@computed_field(description="Path to my step output data")
def my_step_folder(self) -> Path:
    directory = Path(os.path.join(self.root_folder, "my_step"))
    directory.mkdir(exist_ok=True, parents=True)
    return directory
```

### Step 3 — Implement the step class

Create `src/my_step/my_step_pipeline_step.py`:

```python
from src.common.pipeline_steps import MY_STEP
from src.core import BasePipelineStep
from src.utilities.path_utils import is_empty_dir


class MyStepPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(MY_STEP)

    def start(self, dataset_id: str) -> str:
        if is_empty_dir(self._input_directory):
            self._download_input_dataset(dataset_id=dataset_id)

        # Process files in self._input_directory
        # Save results with self._save_locally_data(path, dataframe)

        return self._upload_output_dataset(parent_datasets=[dataset_id])
```

Create `src/my_step/__init__.py`:
```python
from src.my_step.my_step_pipeline_step import MyStepPipelineStep
```

### Step 4 — Register in `pipeline.py`

In `src/pipelines/pipeline.py`:

```python
from src.common.pipeline_steps import MY_STEP
from src.my_step import MyStepPipelineStep


def run_my_step(dataset_id: str) -> str:
    return MyStepPipelineStep().start(dataset_id=dataset_id)


# Inside the PipelineController block:
pipe.add_function_step(
    name=MY_STEP.name,
    task_type=MY_STEP.task_type,
    parents=[PREVIOUS_STEP.name],
    function=run_my_step,
    function_kwargs=dict(dataset_id='${previous_step.dataset_id}'),
    function_return=['dataset_id'],
    cache_executed_step=True,
    continue_behaviour=dict(continue_on_fail=False, continue_on_abort=False),
    time_limit=SETTINGS.clearml.time_limit,
)
```

### Step 5 — Add parameters to `params.yaml`

In `src/params.yaml`, add a section whose key exactly matches `MY_STEP.name`:

```yaml
my_step:
  some_threshold: 0.5
  skip_validation: False
```

Access in step code via `self.step_params.get("some_threshold", 0.5)`.

## Agent Checklist

Before finalizing a new pipeline step:
- [ ] `PipelineStep.name` exactly matches the `params.yaml` top-level key
- [ ] `start()` signature is `(self, dataset_id: str) -> str`
- [ ] Output directory is a `@computed_field` in `StorageSettings`
- [ ] Step is registered in `pipeline.py` with correct `parents=[...]`
- [ ] `function_return=['dataset_id']` is present if a downstream step uses this step's output
- [ ] `src/my_step/__init__.py` exports the step class
