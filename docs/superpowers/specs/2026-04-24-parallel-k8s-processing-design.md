# Parallel Multi-Source Processing via ClearML Queue on Kubernetes

**Date:** 2026-04-24
**Status:** Approved
**Scope:** `copier-python-template/` — parallel file processing, always enabled

---

## Context

The template generates ClearML-tracked ML pipelines for drilling time-series data stored as multiple Excel files pre-uploaded to S3. The current pipeline processes data sequentially. This design adds intra-step parallelism: PREPROCESS and FEATURE_ENGINEER fan out to ClearML worker tasks (one per file) via dedicated K8s queues, then fan back in before passing a single `dataset_id` to the next step.

---

## Pipeline DAG

**Old:** `PRERUN → PREPROCESS → SPLIT_DATASET → FEATURE_ENGINEER → TRAIN`
**New:** `PREPROCESS → FEATURE_ENGINEER → TRAIN`

`PRERUN` and `SPLIT_DATASET` are removed. Files are pre-uploaded to S3 externally. PREPROCESS is the first step and scans S3 for raw files via `StorageSettings`.

---

## Data Flow

```
PREPROCESS (Coordinator) — first step
  ├─ scans S3: {bucket}/{storage.raw_prefix}/ for Excel files
  ├─ creates N ClearML worker tasks (one per file), enqueues to preprocess_queue
  │     Worker task i: reads file_i from S3
  │                  → writes processed/file_i.parquet to S3
  ├─ waits for all tasks; failed workers are logged and isolated
  ├─ merges successful S3 paths → new ClearML Dataset
  └─ → dataset_id

FEATURE_ENGINEER (Coordinator)
  ├─ reads preprocessed S3 paths from ClearML Dataset manifest
  ├─ creates N ClearML worker tasks, enqueues to feature_engineer_queue
  │     Worker task i: reads processed/file_i.parquet from S3
  │                  → downloads labeling config from S3
  │                  → applies labeling for ACCIDENT_TYPE
  │                  → applies feature engineering
  │                  → writes features/file_i.parquet to S3
  ├─ waits for all tasks; failed workers are logged and isolated
  ├─ merges successful S3 paths → new ClearML Dataset
  └─ → dataset_id

TRAIN (unchanged)
  ├─ downloads merged dataset from ClearML
  └─ trains model for the configured ACCIDENT_TYPE
```

**Invariant preserved:** `dataset_id: str` is the only data transfer mechanism between pipeline steps.

**Failure policy:** A failed worker task is logged to the coordinator's ClearML logger and isolated. The coordinator continues with remaining workers and passes only successful outputs to the next step.

---

## Class Hierarchy

```
BasePipelineStep (existing ABC)
└── BaseCoordinatorStep (new ABC)
    ├── PreprocessPipelineStep
    └── FeatureEngineerPipelineStep

BaseWorkerStep (new ABC — standalone, does not inherit BasePipelineStep)
├── PreprocessWorkerStep
└── FeatureEngineerWorkerStep
```

### `BaseCoordinatorStep`

All fan-out / wait / merge logic lives here. Concrete steps implement three methods only.

```python
class BaseCoordinatorStep(BasePipelineStep):

    @abstractmethod
    def worker_entry_point(self) -> str:
        """Path to worker script in repo. e.g. 'src/preprocess/worker.py'"""

    @abstractmethod
    def _queue_name(self) -> str:
        """ClearML queue for this step's workers."""

    def start(self, dataset_id: str | None = None) -> str:
        file_paths = self._list_input_files(dataset_id)
        worker_tasks = self._launch_workers(file_paths)
        successful_paths = self._wait_for_workers(worker_tasks)
        return self._create_output_dataset(dataset_id, successful_paths)

    def _launch_workers(self, file_paths: list[str]) -> list[Task]:
        script = self.task.get_script()
        tasks = []
        for file_path in file_paths:
            worker_task = Task.create(
                project_name=self.task.get_project_name(),
                task_name=f"{self.pipeline_step.name}-{Path(file_path).stem}",
                task_type=TaskTypes.data_processing,
            )
            worker_task.set_repo(
                repo=script["repository"],
                branch=script["branch"],
                commit=script["version_num"],
            )
            worker_task.set_script(entry_point=self.worker_entry_point(), working_dir=".")
            worker_task.set_parent(self.task.id)
            worker_task.connect({
                "input_file_path": file_path,
                "output_path": self._output_path(file_path),
                **self.params.model_dump(),
            })
            Task.enqueue(worker_task, queue_name=self._queue_name())
            tasks.append(worker_task)
        return tasks

    def _wait_for_workers(self, tasks: list[Task]) -> list[str]:
        """Returns output_paths of successfully completed workers only."""
        pending = {t.id: t for t in tasks}
        successful: list[str] = []
        while pending:
            for task_id in list(pending):
                task = Task.get_task(task_id=task_id)
                status = task.get_status()
                if status == Task.TaskStatusEnum.completed:
                    output_path = task.get_parameters()["output_path"]
                    successful.append(output_path)
                    pending.pop(task_id)
                elif status in (Task.TaskStatusEnum.failed, Task.TaskStatusEnum.stopped):
                    self.task.get_logger().report_text(
                        f"Worker failed: {task.name} (id={task_id})"
                    )
                    pending.pop(task_id)
            if pending:
                time.sleep(POLL_INTERVAL_SECONDS)
        return successful
```

`output_storage_path` is derived from `pipeline_step` (already in `BasePipelineStep`) — not redeclared here.

### Concrete coordinator steps

```python
class PreprocessPipelineStep(BaseCoordinatorStep):
    def worker_entry_point(self) -> str: return "src/preprocess/worker.py"
    def _queue_name(self) -> str: return SETTINGS.clearml.preprocess_queue

class FeatureEngineerPipelineStep(BaseCoordinatorStep):
    def worker_entry_point(self) -> str: return "src/features/worker.py"
    def _queue_name(self) -> str: return SETTINGS.clearml.feature_engineer_queue
```

### `BaseWorkerStep`

Standalone ClearML task. No pipeline membership. Picked up by K8s ClearML agent from queue. Linked to coordinator via `set_parent`.

```python
class BaseWorkerStep:
    def __init__(self):
        self.task = Task.init(task_type=TaskTypes.data_processing)
        self.params = self._connect_params()

    @abstractmethod
    def process(self, input_path: str, output_path: str) -> None: ...

    def run(self) -> None:
        self.process(self.params.input_file_path, self.params.output_path)

    @classmethod
    def main(cls) -> None:
        cls().run()
```

Worker is launched as entry point: `python src/preprocess/worker.py`

---

## Settings

```python
class ObjectStorageSettings(BaseModel):
    endpoint: str       # required — e.g. http://minio:9000
    bucket: str         # required
    access_key: str     # required
    secret_key: str     # required
    raw_prefix: str     # required — S3 path to raw Excel files

class StorageSettings(BaseModel):
    # existing fields preserved
    labeling_config_prefix: str = "configs/labeling"
    # full path: {labeling_config_prefix}/{accident_type}.yaml

class ClearmlSettings(BaseModel):
    # existing fields preserved
    preprocess_queue: str              # required — K8s queue for PREPROCESS workers
    feature_engineer_queue: str        # required — K8s queue for FEATURE_ENGINEER workers
    worker_poll_interval_seconds: int = 30  # polling interval when waiting for workers

class Settings(BaseSettings):
    clearml: ClearmlSettings
    storage: StorageSettings = StorageSettings()
    object_storage: ObjectStorageSettings
    accident_type: str           # required — drives labeling config path

    @property
    def labeling_config_path(self) -> str:
        return f"{self.storage.labeling_config_prefix}/{self.accident_type}.yaml"
```

`.env` example:
```bash
OBJECT_STORAGE__ENDPOINT=http://minio:9000
OBJECT_STORAGE__BUCKET=ml-data
OBJECT_STORAGE__ACCESS_KEY=minioadmin
OBJECT_STORAGE__SECRET_KEY=minioadmin
OBJECT_STORAGE__RAW_PREFIX=drilling/raw/

CLEARML__PREPROCESS_QUEUE=preprocess-workers
CLEARML__FEATURE_ENGINEER_QUEUE=feature-engineer-workers

ACCIDENT_TYPE=stuck_pipe
```

---

## File Structure

### New files

```
src/core/coordinator_step.py          # BaseCoordinatorStep ABC
src/core/worker_step.py               # BaseWorkerStep ABC
src/preprocess/worker.py              # PreprocessWorkerStep + entry point
src/features/worker.py                # FeatureEngineerWorkerStep + entry point
```

### Deleted files

```
src/features/split_dataset_pipeline_step.py
```

### Modified files

| File | Change |
|------|--------|
| `src/settings.py.jinja` | Add `ObjectStorageSettings`, `ClearmlSettings` queues, `StorageSettings.labeling_config_prefix`, `Settings.accident_type` |
| `src/common/pipeline_steps.py` | Remove `PRERUN`, `SPLIT_DATASET` constants |
| `src/pipelines/pipeline.py` | New DAG: `PREPROCESS → FEATURE_ENGINEER → TRAIN` |
| `src/preprocess/preprocess_pipeline_step.py` | Inherit `BaseCoordinatorStep` |
| `src/features/feature_engineer_pipeline_step.py` | Inherit `BaseCoordinatorStep` |

### Pipeline DAG in `pipeline.py`

```python
pipe.add_function_step(
    name=PREPROCESS.name,
    function=run_preprocess,
    function_return=["dataset_id"],
)
pipe.add_function_step(
    name=FEATURE_ENGINEER.name,
    function=run_feature_engineer,
    parents=[PREPROCESS.name],
    function_kwargs={"dataset_id": "${preprocess.dataset_id}"},
    function_return=["dataset_id"],
)
pipe.add_function_step(
    name=TRAIN.name,
    function=run_train,
    parents=[FEATURE_ENGINEER.name],
    function_kwargs={"dataset_id": "${feature_engineer.dataset_id}"},
    function_return=["dataset_id"],
)
```

---

## Testing

Tests live in `copier-python-template/tests/` — ClearML and S3 are mocked.

### New test files

```
tests/test_coordinator_step.py
tests/test_worker_step.py
```

### `test_coordinator_step.py`

```python
@patch("clearml.Task.create")
@patch("clearml.Task.enqueue")
def test_launch_workers_creates_one_task_per_file(mock_enqueue, mock_create):
    step = PreprocessPipelineStep(params=PreprocessParams())
    step._launch_workers(["s3://bucket/file1.xlsx", "s3://bucket/file2.xlsx"])
    assert mock_create.call_count == 2
    assert mock_enqueue.call_count == 2

@patch("clearml.Task.get_task")
def test_wait_isolates_failed_workers(mock_get_task):
    # one completed, one failed → only one output_path returned
    ...

def test_output_path_uses_pipeline_step_dir():
    step = PreprocessPipelineStep(params=PreprocessParams())
    path = step._output_path("s3://bucket/raw/file1.xlsx")
    assert "processed" in path
    assert path.endswith("file1.parquet")
```

### `test_worker_step.py`

```python
@patch("boto3.client")
def test_preprocess_worker_reads_input_and_writes_output(mock_s3):
    worker = PreprocessWorkerStep()
    worker.process(
        input_path="s3://bucket/raw/file1.xlsx",
        output_path="s3://bucket/processed/file1.parquet",
    )
    mock_s3.return_value.download_file.assert_called_once()
    mock_s3.return_value.upload_file.assert_called_once()

@patch("boto3.client")
def test_feature_engineer_worker_applies_labeling(mock_s3):
    # verify labeling is applied for the configured accident_type
    ...
```

---

## Agent Checklist

Before modifying `BaseCoordinatorStep` or any coordinator step:
- [ ] Does `_queue_name()` read from `SETTINGS.clearml.*_queue`?
- [ ] Does `set_parent(self.task.id)` appear in every worker task creation?
- [ ] Does `set_repo()` propagate coordinator's git repo + commit to workers?
- [ ] Does a failed worker log to coordinator's ClearML logger and continue (not raise)?
- [ ] Does `start()` still accept `dataset_id: str | None` and return `str`?

Before modifying `BaseWorkerStep` or any worker:
- [ ] Does the worker call `Task.init()` as its first action?
- [ ] Does `process()` read from S3 and write to S3 only?
- [ ] Does `main()` call `cls().run()` as entry point?
