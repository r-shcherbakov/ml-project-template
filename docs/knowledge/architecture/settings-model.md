# Settings Model

## Context

Settings use Pydantic `BaseSettings` to validate configuration and auto-create
required directories. Environment variables override any field using `__` as the
nested delimiter. Storage and artifact directories are created at settings
initialization time — before any pipeline step runs — to guarantee they exist.

## Invariants

1. Nested env var format: `SECTION__FIELD` (double underscore). Example: `CLEARML__EXECUTE_REMOTELY=true`, `CLEARML__QUEUE_NAME=gpu`.
2. New storage directories must be added as `@computed_field` in `StorageSettings`, not created ad-hoc in step code.
3. New artifact directories must be added as `@computed_field` in `ArtifactsSettings`.
4. Directory creation belongs in `@computed_field` methods — never in `BasePipelineStep.start()`.
5. `SETTINGS` is a module-level singleton (`SETTINGS = Settings()` at bottom of `settings.py`). Never instantiate `Settings()` again elsewhere — import `SETTINGS` directly.

## How It Works

```
Environment variables (or .env file)
    ↓ loaded via env_file = os.getenv('ENV', '.env')
Settings(BaseSettings)
    ├── clearml: ClearmlSettings
    │     CLEARML__EXECUTE_REMOTELY (bool, default: False)
    │     CLEARML__QUEUE_NAME       (str,  default: "default")
    │     CLEARML__PROJECT          (str,  default: project_name from template)
    │     CLEARML__TIME_LIMIT       (int,  default: None)
    │
    ├── storage: StorageSettings   ← creates data/ subdirs on init
    │     data/raw/
    │     data/external/
    │     data/processed/
    │     data/splitted/
    │     data/features/
    │     data/prediction/
    │
    ├── artifacts: ArtifactsSettings  ← creates artifacts/ subdirs on init
    │     artifacts/models/
    │     artifacts/reports/
    │     artifacts/plots/
    │
    └── logging: LoggingSettings
          LOGGING__LEVEL (int, default: logging.INFO)
```

Minimum `.env` for local execution: file can be empty (all fields have defaults).
For remote execution: `CLEARML__EXECUTE_REMOTELY=true` and `CLEARML__QUEUE_NAME=<your-queue>`.

Pattern to add a new storage directory:
```python
# In StorageSettings (settings.py):
@computed_field(description="Path to my new data")
def my_new_folder(self) -> Path:
    directory = Path(os.path.join(self.root_folder, "my_new"))
    directory.mkdir(exist_ok=True, parents=True)
    return directory
```

## Agent Checklist

Before adding configuration or directories:
- [ ] Is the new setting a field in the appropriate nested model (`ClearmlSettings`, `StorageSettings`, etc.)?
- [ ] Is the env var name documented as `SECTION__FIELD` format?
- [ ] Is directory creation in a `@computed_field` method, not in step code?
- [ ] Is `SETTINGS` (the singleton) imported and used, not a new `Settings()` instance?
