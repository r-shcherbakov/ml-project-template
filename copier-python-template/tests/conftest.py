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

    class Settings:
        storage = StorageSettings()
        artifacts = ArtifactsSettings()
        clearml = ClearmlSettings()

    stub = types.ModuleType("src.settings")
    stub.StorageSettings = StorageSettings
    stub.ArtifactsSettings = ArtifactsSettings
    stub.Settings = Settings
    stub.SETTINGS = Settings()

    # Make sure src package exists in sys.modules too
    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(_TEMPLATE_ROOT / "src")]
        sys.modules["src"] = src_pkg

    sys.modules["src.settings"] = stub


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


# Register before any test collection so pipeline_steps.py can be imported.
_register_settings_stub()
_register_ml_stubs()
