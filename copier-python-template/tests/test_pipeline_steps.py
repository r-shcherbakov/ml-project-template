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
