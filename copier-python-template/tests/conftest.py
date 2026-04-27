"""
Test configuration for copier-python-template.

These tests run against the template source files in copier-python-template/
before a generated project exists. A minimal src.settings stub is registered
in sys.modules so that pipeline_steps.py and other src modules can be imported
without a live ClearML connection or a rendered settings.py.jinja.
"""
import sys
import types
from pathlib import Path

# Ensure copier-python-template/ is on sys.path so `src.*` imports resolve.
_TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
if str(_TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEMPLATE_ROOT))

_DATA_ROOT = _TEMPLATE_ROOT / "data"


def _make_storage_settings():
    """Return a minimal StorageSettings-like object for testing."""
    class StorageSettings:
        root_folder = _DATA_ROOT

        @property
        def raw_folder(self):
            d = _DATA_ROOT / "raw"
            d.mkdir(exist_ok=True, parents=True)
            return d

        @property
        def processed_folder(self):
            d = _DATA_ROOT / "processed"
            d.mkdir(exist_ok=True, parents=True)
            return d

        @property
        def splitted_folder(self):
            d = _DATA_ROOT / "splitted"
            d.mkdir(exist_ok=True, parents=True)
            return d

        @property
        def features_folder(self):
            d = _DATA_ROOT / "features"
            d.mkdir(exist_ok=True, parents=True)
            return d

        @property
        def prediction_folder(self):
            d = _DATA_ROOT / "prediction"
            d.mkdir(exist_ok=True, parents=True)
            return d

        @property
        def labels_folder(self):
            d = _DATA_ROOT / "labels"
            d.mkdir(exist_ok=True, parents=True)
            return d

        @property
        def external_folder(self):
            d = _DATA_ROOT / "external"
            d.mkdir(exist_ok=True, parents=True)
            return d

    return StorageSettings


def _make_artifacts_settings():
    _ARTIFACTS_ROOT = _TEMPLATE_ROOT / "artifacts"

    class ArtifactsSettings:
        root_folder = _ARTIFACTS_ROOT

        @property
        def models_folder(self):
            d = _ARTIFACTS_ROOT / "models"
            d.mkdir(exist_ok=True, parents=True)
            return d

        @property
        def plots_folder(self):
            d = _ARTIFACTS_ROOT / "plots"
            d.mkdir(exist_ok=True, parents=True)
            return d

        @property
        def reports_folder(self):
            d = _ARTIFACTS_ROOT / "reports"
            d.mkdir(exist_ok=True, parents=True)
            return d

    return ArtifactsSettings


def _register_settings_stub():
    """Register a minimal src.settings stub in sys.modules."""
    StorageSettings = _make_storage_settings()
    ArtifactsSettings = _make_artifacts_settings()

    class ClearmlSettings:
        project = "test-project"
        queue_name = "default"
        execute_remotely = False
        time_limit = None
        tags = ["test-project"]
        worker_poll_interval_seconds = 30

    class ObjectStorageSettings:
        endpoint = "http://localhost:9000"
        bucket = "test-bucket"

        class _Secret:
            def get_secret_value(self):
                return "test-value"

        access_key = _Secret()
        secret_key = _Secret()

    class Settings:
        storage = StorageSettings()
        artifacts = ArtifactsSettings()
        clearml = ClearmlSettings()
        object_storage = ObjectStorageSettings()

    stub = types.ModuleType("src.settings")
    stub.StorageSettings = StorageSettings
    stub.ArtifactsSettings = ArtifactsSettings
    stub.Settings = Settings
    stub.ObjectStorageSettings = ObjectStorageSettings
    stub.SETTINGS = Settings()

    # Make sure src package exists in sys.modules too
    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(_TEMPLATE_ROOT / "src")]
        sys.modules["src"] = src_pkg

    sys.modules["src.settings"] = stub


def _register_preprocess_stub():
    """Stub src.preprocess.preprocess_pipeline_step to expose PreprocessParams
    and PreprocessPipelineStep without triggering the full import chain
    (bottleneck, sklearn.pipeline, etc.)."""
    from pydantic import BaseModel

    class PreprocessParams(BaseModel):
        skip_mark: bool = False

    class PreprocessPipelineStep:
        """Minimal stub — requires params argument (matches real signature)."""
        def __init__(self, params: PreprocessParams):
            self.params = params

    if "src.preprocess" not in sys.modules:
        preprocess_pkg = types.ModuleType("src.preprocess")
        preprocess_pkg.__path__ = [str(_TEMPLATE_ROOT / "src" / "preprocess")]
        sys.modules["src.preprocess"] = preprocess_pkg

    stub = types.ModuleType("src.preprocess.preprocess_pipeline_step")
    stub.PreprocessParams = PreprocessParams
    stub.PreprocessPipelineStep = PreprocessPipelineStep
    sys.modules["src.preprocess.preprocess_pipeline_step"] = stub


def _register_train_stub():
    """Stub src.train.train_pipeline_step to expose TrainParams and
    TrainPipelineStep without triggering the full import chain (catboost, etc.)."""
    from pydantic import BaseModel

    class TrainParams(BaseModel):
        skip_cv: bool = True
        n_splits: int = 4
        train_final_model: bool = True
        binary_threshold: float = 0.5

    class TrainPipelineStep:
        """Minimal stub — requires params argument (matches real signature)."""
        def __init__(self, params: TrainParams):
            self.params = params

    if "src.train" not in sys.modules:
        train_pkg = types.ModuleType("src.train")
        train_pkg.__path__ = [str(_TEMPLATE_ROOT / "src" / "train")]
        sys.modules["src.train"] = train_pkg

    stub = types.ModuleType("src.train.train_pipeline_step")
    stub.TrainParams = TrainParams
    stub.TrainPipelineStep = TrainPipelineStep
    sys.modules["src.train.train_pipeline_step"] = stub


def _register_boto3_stub():
    """Stub boto3 so coordinator_step.py can be imported without boto3 installed."""
    if "boto3" not in sys.modules:
        boto3_stub = types.ModuleType("boto3")

        def _client(*args, **kwargs):
            import unittest.mock as _mock
            return _mock.MagicMock()

        boto3_stub.client = _client
        sys.modules["boto3"] = boto3_stub


def _register_ml_stubs():
    """Stub heavy ML dependencies (pandas, sklearn) so src.core can be
    imported without requiring a full ML installation in the test environment.

    clearml is pre-imported first so it caches the real numpy before we
    register any stubs (clearml internals depend on numpy.ndarray).
    """
    # Pre-import clearml so it loads the real numpy into sys.modules before we
    # register any lightweight stubs.  If clearml is unavailable the tests will
    # fail for a different reason, which is acceptable.
    try:
        import clearml  # noqa: F401
    except Exception:
        pass

    # Stub pandas (not needed by clearml, but needed by pipeline_step.py and
    # src/utilities/utils.py which references pd.DataFrame in function signatures).
    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")

        class _DataFrame:
            pass

        pandas_stub.DataFrame = _DataFrame
        sys.modules["pandas"] = pandas_stub

    # Stub sklearn / sklearn.base for BaseTransformer
    if "sklearn" not in sys.modules:
        sklearn_stub = types.ModuleType("sklearn")
        sklearn_stub.__path__ = []
        sys.modules["sklearn"] = sklearn_stub
    else:
        sklearn_stub = sys.modules["sklearn"]

    if "sklearn.base" not in sys.modules:
        sklearn_base = types.ModuleType("sklearn.base")
        sys.modules["sklearn.base"] = sklearn_base
        sklearn_stub.base = sklearn_base
    else:
        sklearn_base = sys.modules["sklearn.base"]

    if not hasattr(sklearn_base, "BaseEstimator"):
        class BaseEstimator:
            pass
        sklearn_base.BaseEstimator = BaseEstimator

    if not hasattr(sklearn_base, "TransformerMixin"):
        class TransformerMixin:
            pass
        sklearn_base.TransformerMixin = TransformerMixin

    # Stub sklearn.pipeline for preprocess_pipeline_step.py
    if "sklearn.pipeline" not in sys.modules:
        sklearn_pipeline = types.ModuleType("sklearn.pipeline")

        class Pipeline:
            def __init__(self, steps, **kwargs):
                self.steps = steps

        sklearn_pipeline.Pipeline = Pipeline
        sys.modules["sklearn.pipeline"] = sklearn_pipeline
        sklearn_stub.pipeline = sklearn_pipeline

    # Stub sklearn.set_config for preprocess_pipeline_step.py
    if not hasattr(sklearn_stub, "set_config"):
        def set_config(**kwargs):
            pass
        sklearn_stub.set_config = set_config


# Register before any test collection so pipeline_steps.py can be imported.
_register_settings_stub()
_register_ml_stubs()
_register_boto3_stub()
_register_preprocess_stub()
_register_train_stub()
