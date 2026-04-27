from unittest.mock import MagicMock, patch


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


@patch("clearml.Task.enqueue")
@patch("clearml.Task.create")
@patch("clearml.Task.init")
def test_launch_workers_raises_if_no_git_script(mock_init, mock_create, mock_enqueue):
    mock_task = MagicMock()
    mock_task.get_script.return_value = None
    mock_task.id = "coordinator-task-id"
    mock_init.return_value = mock_task

    CoordClass = _make_coordinator_class()
    coord = CoordClass()

    import pytest
    with pytest.raises(RuntimeError, match="no git script info"):
        coord._launch_workers(["s3://bucket/raw/well_001.xlsx"])


@patch("clearml.Task.get_task")
@patch("clearml.Task.init")
def test_wait_for_workers_times_out(mock_init, mock_get_task):
    mock_coord_task = MagicMock()
    mock_init.return_value = mock_coord_task

    mock_running = MagicMock()
    mock_running.get_status.return_value = "in_progress"
    mock_running.name = "preprocess-well_001"
    mock_get_task.return_value = mock_running

    task1 = MagicMock()
    task1.id = "task1"

    CoordClass = _make_coordinator_class()
    coord = CoordClass()
    result = coord._wait_for_workers([task1], timeout_seconds=0.0)

    assert result == []
    mock_coord_task.get_logger.return_value.report_text.assert_called()
