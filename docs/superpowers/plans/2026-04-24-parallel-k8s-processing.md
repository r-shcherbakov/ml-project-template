# Parallel K8s Processing via ClearML Queue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sequential single-file pipeline with a fan-out/fan-in architecture where PREPROCESS and FEATURE_ENGINEER coordinators spawn per-file ClearML worker tasks via K8s queues, merge results into ClearML Datasets, and pass `dataset_id` to the next step unchanged.

**Architecture:** `BaseCoordinatorStep(BasePipelineStep)` handles fan-out (create/enqueue worker tasks), wait (poll ClearML status), and merge (register successful S3 paths in ClearML Dataset). `BaseWorkerStep` is a standalone ClearML task entry point (no pipeline membership) that reads/writes S3. Coordinators propagate their own git repo+commit to each worker task via `set_repo()`.

**Tech Stack:** ClearML SDK, boto3, Pydantic v2, pytest with `unittest.mock`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/settings.py.jinja` | Modify | Add `ObjectStorageSettings`; extend `ClearmlSettings` with worker queues + poll interval; add `StorageSettings.labels_folder`; add `Settings.object_storage`, `accident_type`, `labeling_config_path` |
| `src/common/pipeline_steps.py` | Modify | Remove `PRERUN`, `SPLIT_DATASET`; update `FEATURE_ENGINEER.input_directory` to `processed_folder` |
| `src/core/worker_step.py` | Create | `BaseWorkerStep` ABC — `Task.init`, `_get_worker_params`, abstract `process()`, `run()`, `main()` |
| `src/core/coordinator_step.py` | Create | `BaseCoordinatorStep(BasePipelineStep)` — `_list_input_files`, `_output_path`, `_launch_workers`, `_wait_for_workers`, `_create_output_dataset`, `start()` |
| `src/core/__init__.py` | Modify | Export `BaseCoordinatorStep`, `BaseWorkerStep` |
| `src/preprocess/worker.py` | Create | `PreprocessWorkerStep` — S3 download, sklearn pipeline, S3 upload |
| `src/preprocess/preprocess_pipeline_step.py` | Modify | Inherit `BaseCoordinatorStep`; implement `worker_entry_point()`, `_queue_name()` |
| `src/features/worker.py` | Create | `FeatureEngineerWorkerStep` — S3 download, labeling config, feature engineering, S3 upload |
| `src/features/feature_engineer_pipeline_step.py` | Modify | Inherit `BaseCoordinatorStep`; implement `worker_entry_point()`, `_queue_name()` |
| `src/features/__init__.py` | Modify | Remove `SplitDatasetPipelineStep` export |
| `src/features/split_dataset_pipeline_step.py` | Delete | No longer part of pipeline |
| `src/pipelines/pipeline.py` | Modify | New DAG: `PREPROCESS → FEATURE_ENGINEER → TRAIN`; remove PRERUN + SPLIT_DATASET steps |
| `tests/test_coordinator_step.py` | Create | Coordinator fan-out, wait, isolation tests |
| `tests/test_worker_step.py` | Create | Worker S3 read/write tests |

---

## Task 1: Settings — ObjectStorageSettings + ClearmlSettings + StorageSettings

**Files:**
- Modify: `src/settings.py.jinja`
- Test: `tests/test_settings.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_settings.py
import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError


def test_object_storage_settings_requires_endpoint():
    from src.settings import ObjectStorageSettings
    with pytest.raises(ValidationError):
        ObjectStorageSettings(bucket="b", access_key="k", secret_key="s")


def test_clearml_settings_worker_queues():
    from src.settings import ClearmlSettings
    s = ClearmlSettings(
        preprocess_queue="preprocess-workers",
        feature_engineer_queue="feature-engineer-workers",
    )
    assert s.preprocess_queue == "preprocess-workers"
    assert s.feature_engineer_queue == "feature-engineer-workers"
    assert s.worker_poll_interval_seconds == 30


def test_storage_settings_labels_folder_created():
    from src.settings import StorageSettings
    s = StorageSettings()
    assert s.labels_folder.exists()
    assert s.labels_folder.name == "labels"
    assert s.labels_folder.parent == s.root_folder


def test_settings_labeling_config_path():
    with patch.dict(os.environ, {
        "ACCIDENT_TYPE": "stuck_pipe",
        "OBJECT_STORAGE__ENDPOINT": "http://minio:9000",
        "OBJECT_STORAGE__BUCKET": "ml-data",
        "OBJECT_STORAGE__ACCESS_KEY": "key",
        "OBJECT_STORAGE__SECRET_KEY": "secret",
        "CLEARML__PREPROCESS_QUEUE": "pq",
        "CLEARML__FEATURE_ENGINEER_QUEUE": "fq",
    }):
        from importlib import reload
        import src.settings as settings_module
        reload(settings_module)
        s = settings_module.Settings()
        assert s.labeling_config_path.name == "stuck_pipe.yaml"
        assert s.labeling_config_path.parent == s.storage.labels_folder
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd copier-python-template && poetry run pytest tests/test_settings.py -v
```
Expected: FAIL — `ObjectStorageSettings`, `labels_folder`, `accident_type` not defined yet.

- [ ] **Step 3: Add ObjectStorageSettings to `src/settings.py.jinja`**

After the `create_missed_directory` function, before `ArtifactsSettings`, add:

```python
class ObjectStorageSettings(BaseModel):
    endpoint: str = Field(description="S3/MinIO endpoint URL")
    bucket: str = Field(description="S3 bucket name")
    access_key: str = Field(description="S3 access key")
    secret_key: str = Field(description="S3 secret key")
```

- [ ] **Step 4: Add worker queues + poll interval to `ClearmlSettings`**

In `ClearmlSettings`, after the `time_limit` field, add:

```python
    preprocess_queue: str = Field(description='ClearML queue for PREPROCESS worker tasks')
    feature_engineer_queue: str = Field(description='ClearML queue for FEATURE_ENGINEER worker tasks')
    worker_poll_interval_seconds: int = Field(30, description='Polling interval in seconds when waiting for worker tasks')
```

- [ ] **Step 5: Add `labels_folder` computed field to `StorageSettings`**

In `StorageSettings`, after the `prediction_folder` computed field, add:

```python
    @computed_field(description="Path to the labels")
    def labels_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "labels"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory
```

- [ ] **Step 6: Add `object_storage`, `accident_type`, `labeling_config_path` to `Settings`**

In `Settings`, after `artifacts` field, add:

```python
    object_storage: ObjectStorageSettings = Field(description="S3/MinIO connection settings")
    accident_type: str = Field(description="Accident type, selects labeling config file")

    @property
    def labeling_config_path(self) -> Path:
        return self.storage.labels_folder / f"{self.accident_type}.yaml"
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
poetry run pytest tests/test_settings.py -v
```
Expected: PASS all 4 tests.

- [ ] **Step 8: Commit**

```bash
git add src/settings.py.jinja tests/test_settings.py
git commit -m "feat: add ObjectStorageSettings, worker queues, labels_folder, accident_type"
```

---

## Task 2: Pipeline Steps Registry — Remove PRERUN and SPLIT_DATASET

**Files:**
- Modify: `src/common/pipeline_steps.py`
- Test: `tests/test_pipeline_steps.py` (create)

- [ ] **Step 1: Write failing test**

```python
# tests/test_pipeline_steps.py
import pytest


def test_prerun_removed():
    from src.common import pipeline_steps
    assert not hasattr(pipeline_steps, "PRERUN")


def test_split_dataset_removed():
    from src.common import pipeline_steps
    assert not hasattr(pipeline_steps, "SPLIT_DATASET")


def test_feature_engineer_input_is_processed_folder():
    from src.common.pipeline_steps import FEATURE_ENGINEER
    from src.settings import StorageSettings
    storage = StorageSettings()
    assert FEATURE_ENGINEER.input_directory == storage.processed_folder


def test_preprocess_and_feature_engineer_and_train_exist():
    from src.common.pipeline_steps import PREPROCESS, FEATURE_ENGINEER, TRAIN
    assert PREPROCESS.name == "preprocess"
    assert FEATURE_ENGINEER.name == "feature_engineer"
    assert TRAIN.name == "train"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_pipeline_steps.py -v
```
Expected: FAIL — PRERUN and SPLIT_DATASET still exist, FEATURE_ENGINEER still points to splitted_folder.

- [ ] **Step 3: Update `src/common/pipeline_steps.py`**

Remove the `PRERUN` and `SPLIT_DATASET` constants entirely. Update `FEATURE_ENGINEER.input_directory`:

```python
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

from clearml import TaskTypes

from src.settings import StorageSettings, ArtifactsSettings

storage_settings = StorageSettings()
artifacts_settings = ArtifactsSettings()


@dataclass(frozen=True)
class PipelineStep:
    """Dataclass for describing pipeline steps"""

    name: str
    task_type: str
    input_directory: Optional[Union[str, Path]] = None
    output_directory: Optional[Union[str, Path]] = None

    def __str__(self):
        return self.name


PREPROCESS = PipelineStep(
    name="preprocess",
    task_type=TaskTypes.data_processing.name,
    input_directory=storage_settings.raw_folder,
    output_directory=storage_settings.processed_folder
)
FEATURE_ENGINEER = PipelineStep(
    name="feature_engineer",
    task_type=TaskTypes.data_processing.name,
    input_directory=storage_settings.processed_folder,
    output_directory=storage_settings.features_folder
)
SELECT_FEATURES = PipelineStep(
    name="select_features",
    task_type=TaskTypes.data_processing.name
)
TRAIN = PipelineStep(
    name="train",
    task_type=TaskTypes.training.name,
    input_directory=storage_settings.features_folder,
    output_directory=storage_settings.prediction_folder
)
PLOTTING = PipelineStep(
    name="plotting",
    task_type=TaskTypes.service.name,
    input_directory=storage_settings.prediction_folder,
    output_directory=artifacts_settings.plots_folder
)
HYPERPARAMETER_OPTIMIZATION = PipelineStep(
    name="hyperparameter_optimization",
    task_type=TaskTypes.optimizer.name
)
POSTRUN = PipelineStep(
    name="postrun",
    task_type=TaskTypes.service.name
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_pipeline_steps.py -v
```
Expected: PASS all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/common/pipeline_steps.py tests/test_pipeline_steps.py
git commit -m "refactor: remove PRERUN and SPLIT_DATASET from pipeline steps registry"
```

---

## Task 3: BaseWorkerStep ABC

**Files:**
- Create: `src/core/worker_step.py`
- Test: `tests/test_worker_step.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_worker_step.py
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


def _make_worker_class():
    """Minimal concrete subclass for testing BaseWorkerStep."""
    from src.core.worker_step import BaseWorkerStep

    class ConcreteWorker(BaseWorkerStep):
        def process(self, input_path: str, output_path: str) -> None:
            self.received_input = input_path
            self.received_output = output_path

    return ConcreteWorker


@patch("clearml.Task.init")
def test_worker_step_initializes_clearml_task(mock_init):
    mock_task = MagicMock()
    mock_init.return_value = mock_task
    WorkerClass = _make_worker_class()
    worker = WorkerClass()
    mock_init.assert_called_once()
    assert worker.task == mock_task


@patch("clearml.Task.init")
def test_worker_step_run_calls_process_with_params(mock_init):
    mock_task = MagicMock()
    mock_task.get_parameters.return_value = {
        "worker/input_file_path": "s3://bucket/raw/file.xlsx",
        "worker/output_path": "s3://bucket/processed/file.parquet",
    }
    mock_init.return_value = mock_task
    WorkerClass = _make_worker_class()
    worker = WorkerClass()
    worker.run()
    assert worker.received_input == "s3://bucket/raw/file.xlsx"
    assert worker.received_output == "s3://bucket/processed/file.parquet"


@patch("clearml.Task.init")
def test_worker_step_main_creates_and_runs(mock_init):
    mock_task = MagicMock()
    mock_task.get_parameters.return_value = {
        "worker/input_file_path": "s3://bucket/raw/file.xlsx",
        "worker/output_path": "s3://bucket/processed/file.parquet",
    }
    mock_init.return_value = mock_task
    WorkerClass = _make_worker_class()
    WorkerClass.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_worker_step.py -v
```
Expected: FAIL — `src/core/worker_step.py` does not exist.

- [ ] **Step 3: Create `src/core/worker_step.py`**

```python
# src/core/worker_step.py
from abc import ABC, abstractmethod

from clearml import Task, TaskTypes

from src.settings import SETTINGS


class BaseWorkerStep(ABC):
    """Standalone ClearML task for processing a single file. No pipeline membership."""

    def __init__(self):
        self.task: Task = Task.init(
            project_name=SETTINGS.clearml.project,
            task_name="worker",
            task_type=TaskTypes.data_processing,
            reuse_last_task_id=False,
        )

    def _get_worker_params(self) -> dict:
        return self.task.get_parameters(cast=True)

    @abstractmethod
    def process(self, input_path: str, output_path: str) -> None:
        """Process single file from S3 input_path, save result to S3 output_path."""

    def run(self) -> None:
        params = self._get_worker_params()
        input_path: str = params["worker/input_file_path"]
        output_path: str = params["worker/output_path"]
        self.process(input_path, output_path)

    @classmethod
    def main(cls) -> None:
        cls().run()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_worker_step.py -v
```
Expected: PASS all 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/core/worker_step.py tests/test_worker_step.py
git commit -m "feat: add BaseWorkerStep ABC for per-file S3 processing"
```

---

## Task 4: BaseCoordinatorStep ABC

**Files:**
- Create: `src/core/coordinator_step.py`
- Test: `tests/test_coordinator_step.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_coordinator_step.py
import pytest
from unittest.mock import MagicMock, patch, call
from pydantic import BaseModel


def _make_coordinator_class():
    """Minimal concrete subclass for testing BaseCoordinatorStep."""
    from src.core.coordinator_step import BaseCoordinatorStep
    from src.common.pipeline_steps import PREPROCESS
    from src.preprocess.preprocess_pipeline_step import PreprocessParams

    class ConcreteCoordinator(BaseCoordinatorStep):
        def __init__(self):
            super().__init__(PREPROCESS, params=PreprocessParams())

        def worker_entry_point(self) -> str:
            return "src/preprocess/worker.py"

        def _queue_name(self) -> str:
            return "test-queue"

    return ConcreteCoordinator


@patch("clearml.Task.init")
def test_output_path_uses_pipeline_step_output_directory(mock_init):
    mock_init.return_value = MagicMock()
    CoordClass = _make_coordinator_class()
    coord = CoordClass()
    path = coord._output_path("s3://bucket/raw/well_001.xlsx")
    assert "well_001" in path
    assert path.endswith(".parquet")
    assert "processed" in path


@patch("clearml.Task.enqueue")
@patch("clearml.Task.create")
@patch("clearml.Task.init")
def test_launch_workers_creates_one_task_per_file(mock_init, mock_create, mock_enqueue):
    mock_task = MagicMock()
    mock_task.get_script.return_value = {
        "repository": "https://github.com/org/repo",
        "branch": "main",
        "version_num": "abc123",
    }
    mock_task.get_project_name.return_value = "test-project"
    mock_task.id = "coordinator-task-id"
    mock_init.return_value = mock_task

    mock_worker = MagicMock()
    mock_create.return_value = mock_worker

    CoordClass = _make_coordinator_class()
    coord = CoordClass()
    coord._launch_workers([
        "s3://bucket/raw/well_001.xlsx",
        "s3://bucket/raw/well_002.xlsx",
    ])

    assert mock_create.call_count == 2
    assert mock_enqueue.call_count == 2


@patch("clearml.Task.enqueue")
@patch("clearml.Task.create")
@patch("clearml.Task.init")
def test_launch_workers_sets_parent_on_each_task(mock_init, mock_create, mock_enqueue):
    mock_task = MagicMock()
    mock_task.get_script.return_value = {
        "repository": "https://github.com/org/repo",
        "branch": "main",
        "version_num": "abc123",
    }
    mock_task.get_project_name.return_value = "test-project"
    mock_task.id = "coordinator-task-id"
    mock_init.return_value = mock_task

    mock_worker = MagicMock()
    mock_create.return_value = mock_worker

    CoordClass = _make_coordinator_class()
    coord = CoordClass()
    coord._launch_workers(["s3://bucket/raw/well_001.xlsx"])

    mock_worker.set_parent.assert_called_once_with("coordinator-task-id")


@patch("clearml.Task.get_task")
@patch("clearml.Task.init")
def test_wait_for_workers_returns_only_successful_paths(mock_init, mock_get_task):
    mock_coord_task = MagicMock()
    mock_init.return_value = mock_coord_task

    mock_completed = MagicMock()
    mock_completed.get_status.return_value = "completed"
    mock_completed.get_parameters.return_value = {"worker/output_path": "s3://bucket/processed/well_001.parquet"}
    mock_completed.name = "preprocess-well_001"

    mock_failed = MagicMock()
    mock_failed.get_status.return_value = "failed"
    mock_failed.name = "preprocess-well_002"

    mock_get_task.side_effect = [mock_completed, mock_failed]

    task1 = MagicMock()
    task1.id = "task1"
    task2 = MagicMock()
    task2.id = "task2"

    CoordClass = _make_coordinator_class()
    coord = CoordClass()
    result = coord._wait_for_workers([task1, task2])

    assert result == ["s3://bucket/processed/well_001.parquet"]
    assert len(result) == 1


@patch("clearml.Task.get_task")
@patch("clearml.Task.init")
def test_wait_for_workers_logs_failed_worker(mock_init, mock_get_task):
    mock_coord_task = MagicMock()
    mock_init.return_value = mock_coord_task

    mock_failed = MagicMock()
    mock_failed.get_status.return_value = "failed"
    mock_failed.name = "preprocess-well_001"

    mock_get_task.return_value = mock_failed

    task1 = MagicMock()
    task1.id = "task1"

    CoordClass = _make_coordinator_class()
    coord = CoordClass()
    coord._wait_for_workers([task1])

    mock_coord_task.get_logger.return_value.report_text.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_coordinator_step.py -v
```
Expected: FAIL — `src/core/coordinator_step.py` does not exist.

- [ ] **Step 3: Create `src/core/coordinator_step.py`**

```python
# src/core/coordinator_step.py
import time
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import boto3
from clearml import Dataset, Task, TaskTypes

from src.common.exceptions import PipelineExecutionError
from src.core.pipeline_step import BasePipelineStep
from src.settings import PROJECT_PATH, SETTINGS


class BaseCoordinatorStep(BasePipelineStep):
    """Fan-out/wait/merge coordinator. Concrete steps implement worker_entry_point and _queue_name."""

    @abstractmethod
    def worker_entry_point(self) -> str:
        """Path to worker script in repo. e.g. 'src/preprocess/worker.py'"""

    @abstractmethod
    def _queue_name(self) -> str:
        """ClearML queue name for this step's worker tasks."""

    def _s3_client(self):
        return boto3.client(
            "s3",
            endpoint_url=SETTINGS.object_storage.endpoint,
            aws_access_key_id=SETTINGS.object_storage.access_key,
            aws_secret_access_key=SETTINGS.object_storage.secret_key,
        )

    def _output_path(self, input_s3_path: str) -> str:
        stem = Path(input_s3_path).stem
        bucket = SETTINGS.object_storage.bucket
        output_dir = Path(self._output_directory).relative_to(PROJECT_PATH)
        return f"s3://{bucket}/{output_dir}/{stem}.parquet"

    def _list_input_files(self, dataset_id: Optional[str]) -> list[str]:
        if dataset_id is None:
            return self._list_s3_raw_files()
        return self._list_dataset_files(dataset_id)

    def _list_s3_raw_files(self) -> list[str]:
        s3 = self._s3_client()
        bucket = SETTINGS.object_storage.bucket
        prefix = str(Path(SETTINGS.storage.raw_folder).relative_to(PROJECT_PATH))
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [
            f"s3://{bucket}/{obj['Key']}"
            for obj in response.get("Contents", [])
            if obj["Key"].endswith((".xlsx", ".xls", ".csv"))
        ]

    def _list_dataset_files(self, dataset_id: str) -> list[str]:
        dataset = Dataset.get(dataset_id=dataset_id, only_completed=True)
        return list(dataset.list_files())

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
            worker_task.set_script(
                entry_point=self.worker_entry_point(),
                working_dir=".",
            )
            worker_task.set_parent(self.task.id)
            worker_task.connect(
                {
                    "input_file_path": file_path,
                    "output_path": self._output_path(file_path),
                    **(self.step_params.model_dump() if self.step_params else {}),
                },
                name="worker",
            )
            Task.enqueue(worker_task, queue_name=self._queue_name())
            tasks.append(worker_task)
        return tasks

    def _wait_for_workers(self, tasks: list[Task]) -> list[str]:
        """Returns output_paths of successfully completed workers only."""
        pending = {t.id: t for t in tasks}
        successful: list[str] = []
        while pending:
            for task_id in list(pending):
                remote = Task.get_task(task_id=task_id)
                status = remote.get_status()
                if status == "completed":
                    output_path = remote.get_parameters(cast=True).get("worker/output_path", "")
                    successful.append(output_path)
                    pending.pop(task_id)
                elif status in ("failed", "stopped"):
                    self.task.get_logger().report_text(
                        f"Worker failed: {remote.name} (id={task_id})"
                    )
                    pending.pop(task_id)
            if pending:
                time.sleep(SETTINGS.clearml.worker_poll_interval_seconds)
        return successful

    def _create_output_dataset(self, parent_id: Optional[str], successful_paths: list[str]) -> str:
        parent_datasets = [parent_id] if parent_id else []
        dataset = Dataset.create(
            dataset_project=SETTINGS.clearml.project,
            dataset_name=f"{self.pipeline_step.name.replace('_', ' ')} dataset",
            dataset_tags=SETTINGS.clearml.tags,
            parent_datasets=parent_datasets,
        )
        for path in successful_paths:
            dataset.add_external_files(source_url=path)
        dataset.finalize(auto_upload=True)
        return dataset.id

    def start(self, dataset_id: Optional[str] = None) -> str:
        file_paths = self._list_input_files(dataset_id)
        worker_tasks = self._launch_workers(file_paths)
        successful_paths = self._wait_for_workers(worker_tasks)
        return self._create_output_dataset(dataset_id, successful_paths)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_coordinator_step.py -v
```
Expected: PASS all 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/core/coordinator_step.py tests/test_coordinator_step.py
git commit -m "feat: add BaseCoordinatorStep ABC with fan-out/wait/merge logic"
```

---

## Task 5: Update Exports

**Files:**
- Modify: `src/core/__init__.py`
- Modify: `src/features/__init__.py`

- [ ] **Step 1: Update `src/core/__init__.py`**

```python
from src.core.coordinator_step import BaseCoordinatorStep
from src.core.loader import BaseLoader
from src.core.pipeline_step import BasePipelineStep
from src.core.transformer import BaseTransformer
from src.core.worker_step import BaseWorkerStep
```

- [ ] **Step 2: Update `src/features/__init__.py`**

```python
from src.features.feature_engineer_pipeline_step import FeatureEngineerPipelineStep
```

(Remove `SplitDatasetPipelineStep` import — the file will be deleted in Task 10.)

- [ ] **Step 3: Verify imports work**

```bash
poetry run python -c "from src.core import BaseCoordinatorStep, BaseWorkerStep; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/core/__init__.py src/features/__init__.py
git commit -m "refactor: update core and features exports"
```

---

## Task 6: PreprocessWorkerStep

**Files:**
- Create: `src/preprocess/worker.py`
- Test: `tests/test_preprocess_worker.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocess_worker.py
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytest


@patch("boto3.client")
@patch("clearml.Task.init")
def test_preprocess_worker_downloads_input_from_s3(mock_init, mock_boto):
    mock_task = MagicMock()
    mock_task.get_parameters.return_value = {
        "worker/input_file_path": "s3://ml-data/data/raw/well_001.csv",
        "worker/output_path": "s3://ml-data/data/processed/well_001.parquet",
        "worker/skip_mark": False,
    }
    mock_init.return_value = mock_task

    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3

    with patch("src.utilities.loaders.CsvLoader") as mock_loader, \
         patch("sklearn.pipeline.Pipeline.transform") as mock_transform:
        mock_loader.return_value.load.return_value = pd.DataFrame({"a": [1, 2]})
        mock_transform.return_value = pd.DataFrame({"a": [1, 2]})

        from src.preprocess.worker import PreprocessWorkerStep
        worker = PreprocessWorkerStep()
        worker.process(
            input_path="s3://ml-data/data/raw/well_001.csv",
            output_path="s3://ml-data/data/processed/well_001.parquet",
        )

    mock_s3.download_file.assert_called_once()


@patch("boto3.client")
@patch("clearml.Task.init")
def test_preprocess_worker_uploads_result_to_s3(mock_init, mock_boto):
    mock_task = MagicMock()
    mock_task.get_parameters.return_value = {
        "worker/input_file_path": "s3://ml-data/data/raw/well_001.csv",
        "worker/output_path": "s3://ml-data/data/processed/well_001.parquet",
        "worker/skip_mark": False,
    }
    mock_init.return_value = mock_task

    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3

    with patch("src.utilities.loaders.CsvLoader") as mock_loader, \
         patch("sklearn.pipeline.Pipeline.transform") as mock_transform, \
         patch("pandas.DataFrame.to_parquet"):
        mock_loader.return_value.load.return_value = pd.DataFrame({"a": [1, 2]})
        mock_transform.return_value = pd.DataFrame({"a": [1, 2]})

        from src.preprocess.worker import PreprocessWorkerStep
        worker = PreprocessWorkerStep()
        worker.process(
            input_path="s3://ml-data/data/raw/well_001.csv",
            output_path="s3://ml-data/data/processed/well_001.parquet",
        )

    mock_s3.upload_file.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_preprocess_worker.py -v
```
Expected: FAIL — `src/preprocess/worker.py` does not exist.

- [ ] **Step 3: Create `src/preprocess/worker.py`**

```python
# src/preprocess/worker.py
from pathlib import Path

import boto3
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline

from src.core.worker_step import BaseWorkerStep
from src.preprocess.preprocessor import Preprocessor, MarkDataTransformer
from src.settings import SETTINGS
from src.utilities.loaders import CsvLoader


class PreprocessWorkerStep(BaseWorkerStep):
    def process(self, input_path: str, output_path: str) -> None:
        params = self._get_worker_params()
        skip_mark: bool = params.get("worker/skip_mark", False)
        bucket = SETTINGS.object_storage.bucket

        s3 = boto3.client(
            "s3",
            endpoint_url=SETTINGS.object_storage.endpoint,
            aws_access_key_id=SETTINGS.object_storage.access_key,
            aws_secret_access_key=SETTINGS.object_storage.secret_key,
        )

        local_input = Path("/tmp") / Path(input_path).name
        in_key = input_path.removeprefix(f"s3://{bucket}/")
        s3.download_file(bucket, in_key, str(local_input))

        data = CsvLoader(path=local_input).load()
        steps = [("preprocessor", Preprocessor())]
        if not skip_mark:
            steps.append(("add_target", MarkDataTransformer()))
        set_config(transform_output="pandas")
        pipeline = Pipeline(steps=steps)
        result: pd.DataFrame = pipeline.transform(data)

        local_output = Path("/tmp") / Path(output_path).name
        result.to_parquet(local_output, index=False)
        out_key = output_path.removeprefix(f"s3://{bucket}/")
        s3.upload_file(str(local_output), bucket, out_key)


if __name__ == "__main__":
    PreprocessWorkerStep.main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_preprocess_worker.py -v
```
Expected: PASS both tests.

- [ ] **Step 5: Commit**

```bash
git add src/preprocess/worker.py tests/test_preprocess_worker.py
git commit -m "feat: add PreprocessWorkerStep — S3 download, sklearn pipeline, S3 upload"
```

---

## Task 7: Refactor PreprocessPipelineStep → BaseCoordinatorStep

**Files:**
- Modify: `src/preprocess/preprocess_pipeline_step.py`

The current step processes files sequentially. Replace it entirely with a coordinator that delegates to `PreprocessWorkerStep` via ClearML queue.

- [ ] **Step 1: Write failing test**

```python
# tests/test_preprocess_coordinator.py
from unittest.mock import MagicMock, patch


@patch("clearml.Task.init")
def test_preprocess_step_worker_entry_point(mock_init):
    mock_init.return_value = MagicMock()
    from src.preprocess.preprocess_pipeline_step import PreprocessPipelineStep, PreprocessParams
    step = PreprocessPipelineStep(params=PreprocessParams())
    assert step.worker_entry_point() == "src/preprocess/worker.py"


@patch("clearml.Task.init")
def test_preprocess_step_queue_name_from_settings(mock_init):
    mock_init.return_value = MagicMock()
    from src.preprocess.preprocess_pipeline_step import PreprocessPipelineStep, PreprocessParams
    from src.settings import SETTINGS
    step = PreprocessPipelineStep(params=PreprocessParams())
    assert step._queue_name() == SETTINGS.clearml.preprocess_queue


@patch("clearml.Task.init")
def test_preprocess_step_is_coordinator(mock_init):
    mock_init.return_value = MagicMock()
    from src.preprocess.preprocess_pipeline_step import PreprocessPipelineStep
    from src.core.coordinator_step import BaseCoordinatorStep
    assert issubclass(PreprocessPipelineStep, BaseCoordinatorStep)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_preprocess_coordinator.py -v
```
Expected: FAIL — `PreprocessPipelineStep` is not a `BaseCoordinatorStep` yet.

- [ ] **Step 3: Rewrite `src/preprocess/preprocess_pipeline_step.py`**

```python
# src/preprocess/preprocess_pipeline_step.py
from pydantic import BaseModel

from src.common.pipeline_steps import PREPROCESS
from src.core.coordinator_step import BaseCoordinatorStep
from src.settings import SETTINGS


class PreprocessParams(BaseModel):
    skip_mark: bool = False


class PreprocessPipelineStep(BaseCoordinatorStep):
    def __init__(self, params: PreprocessParams):
        super().__init__(PREPROCESS, params=params)

    def worker_entry_point(self) -> str:
        return "src/preprocess/worker.py"

    def _queue_name(self) -> str:
        return SETTINGS.clearml.preprocess_queue
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_preprocess_coordinator.py tests/test_step_params.py -v
```
Expected: PASS — `PreprocessParams` model tests still pass, coordinator tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/preprocess/preprocess_pipeline_step.py tests/test_preprocess_coordinator.py
git commit -m "refactor: PreprocessPipelineStep now inherits BaseCoordinatorStep"
```

---

## Task 8: FeatureEngineerWorkerStep

**Files:**
- Create: `src/features/worker.py`
- Test: `tests/test_feature_engineer_worker.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_feature_engineer_worker.py
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest


@patch("boto3.client")
@patch("clearml.Task.init")
def test_feature_engineer_worker_downloads_labeling_config(mock_init, mock_boto):
    mock_task = MagicMock()
    mock_task.get_parameters.return_value = {
        "worker/input_file_path": "s3://ml-data/data/processed/well_001.parquet",
        "worker/output_path": "s3://ml-data/data/features/well_001.parquet",
    }
    mock_init.return_value = mock_task

    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3

    with patch("pandas.read_parquet") as mock_read, \
         patch("builtins.open", create=True) as mock_open, \
         patch("yaml.safe_load", return_value={}), \
         patch("src.features.feature_engineer.FeatureEngineer") as mock_fe, \
         patch("pandas.DataFrame.to_parquet"):
        mock_read.return_value = pd.DataFrame({"a": [1]})
        mock_fe.return_value.fit_transform.return_value = pd.DataFrame({"a": [1]})
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value="")

        from src.features.worker import FeatureEngineerWorkerStep
        worker = FeatureEngineerWorkerStep()
        worker.process(
            input_path="s3://ml-data/data/processed/well_001.parquet",
            output_path="s3://ml-data/data/features/well_001.parquet",
        )

    assert mock_s3.download_file.call_count >= 2


@patch("boto3.client")
@patch("clearml.Task.init")
def test_feature_engineer_worker_uploads_result_to_s3(mock_init, mock_boto):
    mock_task = MagicMock()
    mock_task.get_parameters.return_value = {
        "worker/input_file_path": "s3://ml-data/data/processed/well_001.parquet",
        "worker/output_path": "s3://ml-data/data/features/well_001.parquet",
    }
    mock_init.return_value = mock_task

    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3

    with patch("pandas.read_parquet") as mock_read, \
         patch("builtins.open", create=True) as mock_open, \
         patch("yaml.safe_load", return_value={}), \
         patch("src.features.feature_engineer.FeatureEngineer") as mock_fe, \
         patch("pandas.DataFrame.to_parquet"):
        mock_read.return_value = pd.DataFrame({"a": [1]})
        mock_fe.return_value.fit_transform.return_value = pd.DataFrame({"a": [1]})
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)

        from src.features.worker import FeatureEngineerWorkerStep
        worker = FeatureEngineerWorkerStep()
        worker.process(
            input_path="s3://ml-data/data/processed/well_001.parquet",
            output_path="s3://ml-data/data/features/well_001.parquet",
        )

    mock_s3.upload_file.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_feature_engineer_worker.py -v
```
Expected: FAIL — `src/features/worker.py` does not exist.

- [ ] **Step 3: Create `src/features/worker.py`**

```python
# src/features/worker.py
from pathlib import Path

import boto3
import pandas as pd
import yaml

from src.core.worker_step import BaseWorkerStep
from src.features.feature_engineer import FeatureEngineer
from src.settings import PROJECT_PATH, SETTINGS


class FeatureEngineerWorkerStep(BaseWorkerStep):
    def process(self, input_path: str, output_path: str) -> None:
        bucket = SETTINGS.object_storage.bucket
        s3 = boto3.client(
            "s3",
            endpoint_url=SETTINGS.object_storage.endpoint,
            aws_access_key_id=SETTINGS.object_storage.access_key,
            aws_secret_access_key=SETTINGS.object_storage.secret_key,
        )

        local_input = Path("/tmp") / Path(input_path).name
        in_key = input_path.removeprefix(f"s3://{bucket}/")
        s3.download_file(bucket, in_key, str(local_input))

        config_s3_key = str(
            SETTINGS.storage.labels_folder.relative_to(PROJECT_PATH)
        ) + f"/{SETTINGS.accident_type}.yaml"
        local_config = Path("/tmp") / f"{SETTINGS.accident_type}.yaml"
        s3.download_file(bucket, config_s3_key, str(local_config))

        with open(local_config) as f:
            labeling_config = yaml.safe_load(f)

        data = pd.read_parquet(local_input)
        fe = FeatureEngineer(labeling_config=labeling_config)
        result = fe.fit_transform(data)

        local_output = Path("/tmp") / Path(output_path).name
        result.to_parquet(local_output, index=False)
        out_key = output_path.removeprefix(f"s3://{bucket}/")
        s3.upload_file(str(local_output), bucket, out_key)


if __name__ == "__main__":
    FeatureEngineerWorkerStep.main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_feature_engineer_worker.py -v
```
Expected: PASS both tests.

- [ ] **Step 5: Commit**

```bash
git add src/features/worker.py tests/test_feature_engineer_worker.py
git commit -m "feat: add FeatureEngineerWorkerStep — S3 download, labeling config, feature engineering"
```

---

## Task 9: Refactor FeatureEngineerPipelineStep → BaseCoordinatorStep

**Files:**
- Modify: `src/features/feature_engineer_pipeline_step.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_feature_engineer_coordinator.py
from unittest.mock import MagicMock, patch


@patch("clearml.Task.init")
def test_feature_engineer_step_worker_entry_point(mock_init):
    mock_init.return_value = MagicMock()
    from src.features.feature_engineer_pipeline_step import (
        FeatureEngineerPipelineStep, FeatureEngineerParams
    )
    step = FeatureEngineerPipelineStep(params=FeatureEngineerParams())
    assert step.worker_entry_point() == "src/features/worker.py"


@patch("clearml.Task.init")
def test_feature_engineer_step_queue_name_from_settings(mock_init):
    mock_init.return_value = MagicMock()
    from src.features.feature_engineer_pipeline_step import (
        FeatureEngineerPipelineStep, FeatureEngineerParams
    )
    from src.settings import SETTINGS
    step = FeatureEngineerPipelineStep(params=FeatureEngineerParams())
    assert step._queue_name() == SETTINGS.clearml.feature_engineer_queue


@patch("clearml.Task.init")
def test_feature_engineer_step_is_coordinator(mock_init):
    mock_init.return_value = MagicMock()
    from src.features.feature_engineer_pipeline_step import FeatureEngineerPipelineStep
    from src.core.coordinator_step import BaseCoordinatorStep
    assert issubclass(FeatureEngineerPipelineStep, BaseCoordinatorStep)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_feature_engineer_coordinator.py -v
```
Expected: FAIL — `FeatureEngineerPipelineStep` is not a `BaseCoordinatorStep` yet.

- [ ] **Step 3: Rewrite `src/features/feature_engineer_pipeline_step.py`**

```python
# src/features/feature_engineer_pipeline_step.py
from pydantic import BaseModel

from src.common.pipeline_steps import FEATURE_ENGINEER
from src.core.coordinator_step import BaseCoordinatorStep
from src.settings import SETTINGS


class FeatureEngineerParams(BaseModel):
    pass


class FeatureEngineerPipelineStep(BaseCoordinatorStep):
    def __init__(self, params: FeatureEngineerParams = None):
        super().__init__(FEATURE_ENGINEER, params=params or FeatureEngineerParams())

    def worker_entry_point(self) -> str:
        return "src/features/worker.py"

    def _queue_name(self) -> str:
        return SETTINGS.clearml.feature_engineer_queue
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_feature_engineer_coordinator.py -v
```
Expected: PASS all 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/features/feature_engineer_pipeline_step.py tests/test_feature_engineer_coordinator.py
git commit -m "refactor: FeatureEngineerPipelineStep now inherits BaseCoordinatorStep"
```

---

## Task 10: Delete SplitDataset + Rewrite Pipeline DAG

**Files:**
- Delete: `src/features/split_dataset_pipeline_step.py`
- Modify: `src/pipelines/pipeline.py`

- [ ] **Step 1: Delete `split_dataset_pipeline_step.py`**

```bash
rm copier-python-template/src/features/split_dataset_pipeline_step.py
```

- [ ] **Step 2: Rewrite `src/pipelines/pipeline.py`**

```python
# src/pipelines/pipeline.py
from clearml import PipelineController

from src.common.pipeline_steps import PREPROCESS, FEATURE_ENGINEER, TRAIN
from src.features.feature_engineer_pipeline_step import FeatureEngineerPipelineStep, FeatureEngineerParams
from src.preprocess.preprocess_pipeline_step import PreprocessPipelineStep, PreprocessParams
from src.train.train_pipeline_step import TrainPipelineStep, TrainParams
from src.settings import SETTINGS


def run_preprocess_step() -> str:
    return PreprocessPipelineStep(
        params=PreprocessParams(skip_mark=False),
    ).start(dataset_id=None)


def run_feature_engineer_step(dataset_id: str) -> str:
    return FeatureEngineerPipelineStep(
        params=FeatureEngineerParams(),
    ).start(dataset_id=dataset_id)


def run_train_step(dataset_id: str) -> None:
    return TrainPipelineStep(
        params=TrainParams(
            skip_cv=True,
            n_splits=4,
            train_final_model=True,
            binary_threshold=0.5,
        ),
    ).start(dataset_id=dataset_id)


if __name__ == '__main__':
    pipe = PipelineController(
        name=f'{SETTINGS.clearml.project} pipeline',
        project=SETTINGS.clearml.project,
        target_project=SETTINGS.clearml.project,
        add_pipeline_tags=False,
        auto_version_bump=True,
        add_run_number=False,
        packages="./requirements.txt",
    )

    pipe.add_function_step(
        name=PREPROCESS.name,
        task_type=PREPROCESS.task_type,
        function=run_preprocess_step,
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=FEATURE_ENGINEER.name,
        task_type=FEATURE_ENGINEER.task_type,
        parents=[PREPROCESS.name],
        function=run_feature_engineer_step,
        function_kwargs=dict(
            dataset_id='${preprocess.dataset_id}',
        ),
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=TRAIN.name,
        task_type=TRAIN.task_type,
        parents=[FEATURE_ENGINEER.name],
        function=run_train_step,
        function_kwargs=dict(
            dataset_id='${feature_engineer.dataset_id}',
        ),
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.set_default_execution_queue(SETTINGS.clearml.queue_name)
    if SETTINGS.clearml.execute_remotely:
        pipe.start(queue=SETTINGS.clearml.queue_name)
    else:
        pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline finished")
```

- [ ] **Step 3: Run full test suite**

```bash
poetry run pytest tests/ -v
```
Expected: PASS — no references to PRERUN, SPLIT_DATASET, or SplitDatasetPipelineStep.

- [ ] **Step 4: Verify pipeline imports cleanly**

```bash
poetry run python -c "import src.pipelines.pipeline; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/pipeline.py
git rm src/features/split_dataset_pipeline_step.py
git commit -m "refactor: rewrite pipeline DAG — PREPROCESS → FEATURE_ENGINEER → TRAIN"
```

---

## Known Follow-up

`TrainPipelineStep` currently expects `train.gz` / `test.gz` (pickle) files from `SplitDatasetPipelineStep`. After this plan, FEATURE_ENGINEER outputs `.parquet` files per source file registered as ClearML Dataset external files. TRAIN will need a follow-up task to:
1. Download external `.parquet` files via `get_mutable_local_copy()`
2. Concatenate them into a single DataFrame
3. Update the `_get_data()` method to read `.parquet` instead of pickle

Create a separate beads issue for this before closing the epic.
