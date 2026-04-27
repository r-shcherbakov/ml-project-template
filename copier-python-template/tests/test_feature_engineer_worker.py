"""Tests for FeatureEngineerWorkerStep.

The _build_feature_engineer builder function is patched via patch.object so we
avoid importing the heavy sklearn/numpy chain in the test environment.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch


def _get_worker_module():
    """Import src.features.worker and return the module, ensuring the
    src.features package is present in sys.modules first."""
    if "src.features" not in sys.modules:
        features_pkg = types.ModuleType("src.features")
        features_pkg.__path__ = [
            str(Path(__file__).resolve().parents[1] / "src" / "features")
        ]
        sys.modules["src.features"] = features_pkg

    import src.features.worker as worker_mod
    return worker_mod


def _run_feature_engineer_worker(mock_task, mock_boto):
    """Run FeatureEngineerWorkerStep.process() with standard test inputs."""
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3
    mock_task.get_parameters.return_value = {
        "worker/input_file_path": "s3://ml-data/data/processed/well_001.parquet",
        "worker/output_path": "s3://ml-data/data/features/well_001.parquet",
    }

    worker_mod = _get_worker_module()
    mock_df = MagicMock()
    mock_fe = MagicMock()
    mock_fe.fit_transform.return_value = mock_df

    with patch("pandas.read_parquet", return_value=mock_df, create=True), \
         patch("yaml.safe_load", return_value={}, create=True), \
         patch("builtins.open", MagicMock()), \
         patch.object(worker_mod, "_build_feature_engineer", return_value=mock_fe):
        worker = worker_mod.FeatureEngineerWorkerStep()
        worker.process(
            input_path="s3://ml-data/data/processed/well_001.parquet",
            output_path="s3://ml-data/data/features/well_001.parquet",
        )
    return mock_s3


@patch("boto3.client")
@patch("clearml.Task.init")
def test_feature_engineer_worker_downloads_labeling_config(mock_init, mock_boto):
    mock_task = MagicMock()
    mock_init.return_value = mock_task
    mock_s3 = _run_feature_engineer_worker(mock_task, mock_boto)
    # Must download both input parquet and labeling config
    assert mock_s3.download_file.call_count >= 2


@patch("boto3.client")
@patch("clearml.Task.init")
def test_feature_engineer_worker_uploads_result_to_s3(mock_init, mock_boto):
    mock_task = MagicMock()
    mock_init.return_value = mock_task
    mock_s3 = _run_feature_engineer_worker(mock_task, mock_boto)
    mock_s3.upload_file.assert_called_once()
