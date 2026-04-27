"""Tests for PreprocessWorkerStep.

Builder functions (_build_preprocessor, _build_mark_transformer) are patched
via patch.object so we avoid importing the heavy sklearn/bottleneck chain in
the test environment.
"""
import sys
import types
from unittest.mock import MagicMock, patch

def _get_worker_module():
    """Import src.preprocess.worker and return the module, ensuring the
    src.preprocess package is present in sys.modules first."""
    if "src.preprocess" not in sys.modules:
        preprocess_pkg = types.ModuleType("src.preprocess")
        sys.modules["src.preprocess"] = preprocess_pkg

    import src.preprocess.worker as worker_mod
    return worker_mod


def _run_worker(mock_task, mock_boto):
    """Run PreprocessWorkerStep.process() with standard test inputs."""
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3
    mock_task.get_parameters.return_value = {
        "worker/input_file_path": "s3://ml-data/data/raw/well_001.csv",
        "worker/output_path": "s3://ml-data/data/processed/well_001.parquet",
        "worker/skip_mark": False,
    }

    worker_mod = _get_worker_module()
    mock_df = MagicMock()
    with patch("pandas.read_csv", return_value=mock_df, create=True), \
         patch("sklearn.pipeline.Pipeline.transform", return_value=mock_df, create=True), \
         patch.object(worker_mod, "_build_preprocessor", return_value=MagicMock()), \
         patch.object(worker_mod, "_build_mark_transformer", return_value=MagicMock()):
        worker = worker_mod.PreprocessWorkerStep()
        worker.process(
            input_path="s3://ml-data/data/raw/well_001.csv",
            output_path="s3://ml-data/data/processed/well_001.parquet",
        )
    return mock_s3


@patch("boto3.client")
@patch("clearml.Task.init")
def test_preprocess_worker_downloads_input_from_s3(mock_init, mock_boto):
    mock_task = MagicMock()
    mock_init.return_value = mock_task
    mock_s3 = _run_worker(mock_task, mock_boto)
    mock_s3.download_file.assert_called_once()


@patch("boto3.client")
@patch("clearml.Task.init")
def test_preprocess_worker_uploads_result_to_s3(mock_init, mock_boto):
    mock_task = MagicMock()
    mock_init.return_value = mock_task
    mock_s3 = _run_worker(mock_task, mock_boto)
    mock_s3.upload_file.assert_called_once()
