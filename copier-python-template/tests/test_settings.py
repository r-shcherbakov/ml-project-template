# tests/test_settings.py
import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError


def test_object_storage_settings_requires_endpoint():
    from src.settings import ObjectStorageSettings
    with pytest.raises(ValidationError):
        ObjectStorageSettings(bucket="b", access_key="k", secret_key="s")


def test_clearml_settings_worker_queues():
    from src.settings import ClearmlSettings
    s = ClearmlSettings(
        preprocess_queue="preprocess-workers",
        feature_engineer_queue="feature-engineer-workers",
    )
    assert s.preprocess_queue == "preprocess-workers"
    assert s.feature_engineer_queue == "feature-engineer-workers"
    assert s.worker_poll_interval_seconds == 30


def test_storage_settings_labels_folder_created():
    from src.settings import StorageSettings
    s = StorageSettings()
    assert s.labels_folder.exists()
    assert s.labels_folder.name == "labels"
    assert s.labels_folder.parent == s.root_folder


def test_settings_labeling_config_path():
    with patch.dict(os.environ, {
        "ACCIDENT_TYPE": "stuck_pipe",
        "OBJECT_STORAGE__ENDPOINT": "http://minio:9000",
        "OBJECT_STORAGE__BUCKET": "ml-data",
        "OBJECT_STORAGE__ACCESS_KEY": "key",
        "OBJECT_STORAGE__SECRET_KEY": "secret",
        "CLEARML__PREPROCESS_QUEUE": "pq",
        "CLEARML__FEATURE_ENGINEER_QUEUE": "fq",
    }):
        from importlib import reload
        import src.settings as settings_module
        reload(settings_module)
        s = settings_module.Settings()
        assert s.labeling_config_path.name == "stuck_pipe.yaml"
        assert s.labeling_config_path.parent == s.storage.labels_folder
