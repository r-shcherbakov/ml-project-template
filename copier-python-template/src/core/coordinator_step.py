# -*- coding: utf-8 -*-
"""Base class for fan-out/wait/merge coordinator pipeline steps."""
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import boto3
from clearml import Dataset, Task, TaskTypes

from src.core.pipeline_step import BasePipelineStep
from src.settings import SETTINGS


class BaseCoordinatorStep(BasePipelineStep):
    """Fan-out/wait/merge coordinator. Concrete steps implement worker_entry_point and _queue_name."""

    @abstractmethod
    def worker_entry_point(self) -> str:
        """Path to worker script in repo. e.g. 'src/preprocess/worker.py'"""

    @abstractmethod
    def _queue_name(self) -> str:
        """ClearML queue name for this step's worker tasks."""

    def _s3_client(self) -> Any:
        return boto3.client(
            "s3",
            endpoint_url=SETTINGS.object_storage.endpoint,
            aws_access_key_id=SETTINGS.object_storage.access_key.get_secret_value(),
            aws_secret_access_key=SETTINGS.object_storage.secret_key.get_secret_value(),
        )

    def _output_path(self, input_s3_path: str) -> str:
        stem = Path(input_s3_path).stem
        bucket = SETTINGS.object_storage.bucket
        output_dir = Path(self.pipeline_step.output_directory).name
        return f"s3://{bucket}/{output_dir}/{stem}.parquet"

    def _list_input_files(self, dataset_id: Optional[str]) -> list[str]:
        if dataset_id is None:
            return self._list_s3_raw_files()
        return self._list_dataset_files(dataset_id)

    def _list_s3_raw_files(self) -> list[str]:
        s3 = self._s3_client()
        bucket = SETTINGS.object_storage.bucket
        prefix = Path(SETTINGS.storage.raw_folder).name
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
        if not script or not script.get("repository"):
            raise RuntimeError(
                "Coordinator task has no git script info; "
                "cannot propagate repo to worker tasks."
            )
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

    def _wait_for_workers(
        self, tasks: list[Task], timeout_seconds: float = 3600.0
    ) -> list[str]:
        """Returns output_paths of successfully completed workers only."""
        deadline = time.monotonic() + timeout_seconds
        pending = {t.id: t for t in tasks}
        successful: list[str] = []
        while pending:
            if time.monotonic() > deadline:
                for task_id in list(pending):
                    self.task.get_logger().report_text(
                        f"Worker timed out: {task_id}"
                    )
                break
            for task_id in list(pending):
                remote = Task.get_task(task_id=task_id)
                status = remote.get_status()
                if status == "completed":
                    output_path = remote.get_parameters(cast=True).get("worker/output_path", "")
                    if output_path:
                        successful.append(output_path)
                    else:
                        self.task.get_logger().report_text(
                            f"Worker completed but reported no output_path: {remote.name}"
                        )
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
