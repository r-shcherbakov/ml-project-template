# tests/test_settings.py
import pytest
from pydantic import ValidationError


def test_object_storage_settings_requires_endpoint():
    from src.settings import ObjectStorageSettings
    with pytest.raises(ValidationError):
        ObjectStorageSettings(bucket="b", access_key="k", secret_key="s")


def test_clearml_settings_worker_queues():
    from src.settings import ClearmlSettings
    # defaults are None when not set
    s_default = ClearmlSettings()
    assert s_default.preprocess_queue is None
    assert s_default.feature_engineer_queue is None
    assert s_default.worker_poll_interval_seconds == 30
    # can be set explicitly
    s = ClearmlSettings(
        preprocess_queue="preprocess-workers",
        feature_engineer_queue="feature-engineer-workers",
    )
    assert s.preprocess_queue == "preprocess-workers"
    assert s.feature_engineer_queue == "feature-engineer-workers"


def test_storage_settings_labels_folder_created():
    from src.settings import StorageSettings
    s = StorageSettings()
    assert s.labels_folder.exists()
    assert s.labels_folder.name == "labels"
    assert s.labels_folder.parent == s.root_folder


def test_settings_labeling_config_path():
    from src.settings import Settings
    s = Settings(
        accident_type="stuck_pipe",
        object_storage={
            "endpoint": "http://minio:9000",
            "bucket": "ml-data",
            "access_key": "key",
            "secret_key": "secret",
        },
        clearml={"preprocess_queue": "pq", "feature_engineer_queue": "fq"},
    )
    assert s.labeling_config_path.name == "stuck_pipe.yaml"
    assert s.labeling_config_path.parent == s.storage.labels_folder
