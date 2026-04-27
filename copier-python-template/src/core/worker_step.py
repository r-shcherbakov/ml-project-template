# -*- coding: utf-8 -*-
"""Base class for standalone ClearML worker tasks."""
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

    def _get_worker_params(self) -> dict[str, str]:
        return self.task.get_parameters(cast=True)

    @abstractmethod
    def process(self, input_path: str, output_path: str) -> None:
        """Process single file from S3 input_path, save result to S3 output_path."""

    def run(self) -> None:
        try:
            params = self._get_worker_params()
            input_path: str = params["worker/input_file_path"]
            output_path: str = params["worker/output_path"]
        except KeyError as e:
            self.task.get_logger().report_text(f"Missing required worker parameter: {e}")
            raise ValueError(f"Missing required worker parameter: {e}") from e
        self._worker_params: dict[str, str] = params
        self.process(input_path, output_path)

    @classmethod
    def main(cls) -> None:
        cls().run()
