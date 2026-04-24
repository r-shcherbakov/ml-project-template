"""Unit tests for pipeline step Params models."""
import pytest
from pydantic import ValidationError

from src.preprocess.preprocess_pipeline_step import PreprocessParams, PreprocessPipelineStep
from src.features.split_dataset_pipeline_step import SplitDatasetParams, SplitDatasetPipelineStep
from src.train.train_pipeline_step import TrainParams, TrainPipelineStep


def test_preprocess_params_defaults():
    p = PreprocessParams()
    assert p.skip_mark is False


def test_preprocess_params_override():
    p = PreprocessParams(skip_mark=True)
    assert p.skip_mark is True


def test_split_dataset_params_defaults():
    p = SplitDatasetParams()
    assert p.split_test is True
    assert p.num_test_objects == 2
    assert p.test_objects is None


def test_train_params_defaults():
    p = TrainParams()
    assert p.skip_cv is True
    assert p.n_splits == 4
    assert p.train_final_model is True
    assert p.binary_threshold == 0.5


def test_train_params_type_validation():
    with pytest.raises(ValidationError):
        TrainParams(n_splits="not_an_int")


def test_preprocess_step_requires_params():
    with pytest.raises(TypeError):
        PreprocessPipelineStep()


def test_split_dataset_step_requires_params():
    with pytest.raises(TypeError):
        SplitDatasetPipelineStep()


def test_train_step_requires_params():
    with pytest.raises(TypeError):
        TrainPipelineStep()
