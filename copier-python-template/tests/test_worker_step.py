from unittest.mock import MagicMock, patch


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
    mock_init.assert_called_once()
