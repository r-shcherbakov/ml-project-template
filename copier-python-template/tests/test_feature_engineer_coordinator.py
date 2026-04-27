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
    step = FeatureEngineerPipelineStep(params=FeatureEngineerParams())
    assert step._queue_name() == "feature-engineer-workers"


@patch("clearml.Task.init")
def test_feature_engineer_step_is_coordinator(mock_init):
    mock_init.return_value = MagicMock()
    from src.features.feature_engineer_pipeline_step import FeatureEngineerPipelineStep
    from src.core.coordinator_step import BaseCoordinatorStep
    assert issubclass(FeatureEngineerPipelineStep, BaseCoordinatorStep)
