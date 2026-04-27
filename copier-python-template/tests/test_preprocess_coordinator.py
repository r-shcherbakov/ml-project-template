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
    step = PreprocessPipelineStep(params=PreprocessParams())
    assert step._queue_name() == "preprocess-workers"


@patch("clearml.Task.init")
def test_preprocess_step_is_coordinator(mock_init):
    mock_init.return_value = MagicMock()
    from src.preprocess.preprocess_pipeline_step import PreprocessPipelineStep
    from src.core.coordinator_step import BaseCoordinatorStep
    assert issubclass(PreprocessPipelineStep, BaseCoordinatorStep)
