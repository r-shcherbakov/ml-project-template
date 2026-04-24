# Parallel Multi-Source Processing via ClearML Queue on Kubernetes

**Date:** 2026-04-24  
**Status:** Approved  
**Scope:** `copier-python-template/` — new capability, activated via `parallel_processing: bool` copier variable

---

## Context

The template generates ClearML-tracked ML pipelines for drilling time-series data stored as multiple Excel files. The current pipeline processes a single dataset sequentially. This design adds intra-step parallelism: each processing step fans out across Kubernetes pods (one per file), then fans back in before passing a single `dataset_id` to the next step.

---

## Pipeline DAG

Old: `PRERUN → PREPROCESS → SPLIT_DATASET → FEATURE_ENGINEER → TRAIN`  
New: `PRERUN → PREPROCESS → FEATURE_ENGINEER → MERGE → TRAIN`

`SPLIT_DATASET` is removed. `MERGE` is added between `FEATURE_ENGINEER` and `TRAIN`.

---

## Data Flow

```
PRERUN (ClearML Task)
  ├─ reads Excel files from raw data directory
  ├─ uploads each file to object storage (MinIO/S3)
  ├─ registers MinIO paths in ClearML Dataset (manifest only, no data in ClearML)
  └─ → dataset_id

PREPROCESS Coordinator (ClearML Task, K8s pod via ClearML queue)
  ├─ reads MinIO file paths from ClearML Dataset manifest (no data download)
  ├─ creates up to max_concurrent_pods Worker Pods via K8s API
  │     Worker Pod i: reads file_i from source → writes to StorageSettings.processed/file_i.parquet
  ├─ waits for all pods (rolling window, failure isolation)
  ├─ registers successful output paths in ClearML Dataset
  └─ → dataset_id

FEATURE_ENGINEER Coordinator (same pattern as PREPROCESS)
  ├─ reads preprocessed MinIO paths from ClearML Dataset manifest
  ├─ creates Worker Pods → writes to StorageSettings.features/file_i.parquet
  └─ → dataset_id

MERGE (ClearML Task)
  ├─ reads feature MinIO paths from ClearML Dataset manifest
  ├─ downloads all files from object storage
  ├─ concatenates into one DataFrame
  ├─ uploads as single ClearML Dataset
  └─ → dataset_id

TRAIN (ClearML Task, unchanged)
  ├─ downloads merged dataset from ClearML
  └─ trains one model
```

**Invariant preserved:** `dataset_id: str` remains the only data transfer mechanism between pipeline steps (ClearML contract unchanged).

---

## Class Hierarchy

```
BasePipelineStep (existing ABC)
├── BaseCoordinatorStep (NEW ABC)
│   ├── PreprocessPipelineStep
│   └── FeatureEngineerPipelineStep
├── BaseMergeStep (NEW)
│   └── MergePipelineStep
└── TrainPipelineStep (unchanged)

BaseWorkerStep (NEW ABC — no ClearML, no BasePipelineStep)
├── PreprocessWorkerStep
└── FeatureEngineerWorkerStep
```

### `BaseCoordinatorStep`

```python
class BaseCoordinatorStep(BasePipelineStep):

    @abstractmethod
    def worker_module(self) -> str:
        """Python module path. e.g. 'src.preprocess.worker'"""

    @abstractmethod
    def worker_settings(self) -> WorkerResourceSettings:
        """e.g. return SETTINGS.preprocess.worker"""

    @abstractmethod
    def output_storage_path(self) -> Path:
        """Derived from StorageSettings. e.g. return SETTINGS.storage.processed"""

    def _output_path(self, file_path: str) -> str:
        stem = Path(file_path).stem
        return str(self.output_storage_path() / f"{stem}.parquet")

    def start(self, dataset_id: str) -> str:
        file_paths = self._list_dataset_files(dataset_id)
        output_paths = [self._output_path(fp) for fp in file_paths]
        successful_outputs = self._run_workers(file_paths, output_paths)
        return self._create_output_dataset(dataset_id, successful_outputs)
```

Concrete coordinators implement only the three abstract methods:

```python
class PreprocessPipelineStep(BaseCoordinatorStep):
    def worker_module(self) -> str: return "src.preprocess.worker"
    def worker_settings(self) -> WorkerResourceSettings: return SETTINGS.preprocess.worker
    def output_storage_path(self) -> Path: return SETTINGS.storage.processed
```

### `BaseWorkerStep`

Standalone class. No ClearML, no `BasePipelineStep`. Entry point for K8s worker pods.

```python
class BaseWorkerStep:
    @abstractmethod
    def process(self, file_path: Path, params: BaseModel) -> pd.DataFrame: ...

    def run(self) -> None:
        file_path = Path(os.environ["INPUT_FILE_PATH"])
        output_path = os.environ["OUTPUT_PATH"]
        params = self._load_params()
        df = self.process(file_path, params)
        self._save_to_object_storage(df, output_path)

    @classmethod
    def main(cls) -> None:
        cls().run()
```

### `BaseMergeStep`

```python
class BaseMergeStep(BasePipelineStep):
    def start(self, dataset_id: str) -> str:
        paths = self._list_dataset_files(dataset_id)
        frames = [self._download_from_storage(p) for p in paths]
        merged = pd.concat(frames, ignore_index=True)
        return self._upload_dataset(merged, parent_id=dataset_id)
```

---

## K8s Pod Lifecycle

### Rolling window with failure isolation

```python
def _run_workers(
    self, file_paths: list[str], output_paths: list[str]
) -> list[str]:
    """Returns output_paths of successfully completed workers only."""
    api = kubernetes.client.CoreV1Api()
    queue = list(zip(file_paths, output_paths))
    active: dict[str, str] = {}   # pod_name → output_path
    successful: list[str] = []
    deadline = time.time() + SETTINGS.k8s.pod_timeout_seconds

    while queue or active:
        if time.time() > deadline:
            raise TimeoutError(f"Worker batch timed out, still active: {list(active)}")

        while queue and len(active) < SETTINGS.k8s.max_concurrent_pods:
            file_path, output_path = queue.pop(0)
            pod_name = self._create_worker_pod(api, file_path, output_path)
            active[pod_name] = output_path

        time.sleep(SETTINGS.k8s.pod_polling_interval_seconds)

        for pod_name in list(active):
            phase = api.read_namespaced_pod(
                pod_name, SETTINGS.k8s.namespace
            ).status.phase
            if phase == "Succeeded":
                successful.append(active.pop(pod_name))
            elif phase == "Failed":
                active.pop(pod_name)
                self._log_pod_failure(api, pod_name)

    return successful
```

**Failure policy:** A failed pod is isolated — its error is logged to ClearML Logger (`task.get_logger().report_text()`), pod logs are included, and processing continues. The coordinator returns a dataset containing only successful outputs. A total timeout raises `TimeoutError` and fails the coordinator.

### Pod spec

```python
V1Pod(
    spec=V1PodSpec(
        restart_policy="Never",
        service_account_name=SETTINGS.k8s.service_account,
        containers=[V1Container(
            image=SETTINGS.k8s.worker_image,
            command=["python", "-m", self.worker_module()],
            env=[
                V1EnvVar("INPUT_FILE_PATH", file_path),
                V1EnvVar("OUTPUT_PATH", output_path),
                V1EnvVar("STEP_PARAMS_JSON", self.step_params.model_dump_json()),
            ],
            resources=V1ResourceRequirements(
                requests={"cpu": ws.cpu_request, "memory": ws.memory_request},
                limits={"cpu": ws.cpu_limit, "memory": ws.memory_limit},
            ),
        )],
    ),
)
```

Pod name: `{step_name}-{file_stem}` (lowercase, hyphens).

---

## Settings

All settings follow the existing `__` delimiter pattern from `src/settings.py`.

```python
class WorkerResourceSettings(BaseModel):
    cpu_request: str = "1"
    cpu_limit: str = "2"
    memory_request: str = "2Gi"
    memory_limit: str = "4Gi"

class K8sSettings(BaseModel):
    namespace: str = "default"
    service_account: str = "clearml-worker"
    worker_image: str                         # required, no default
    max_concurrent_pods: int = 5
    pod_polling_interval_seconds: int = 10
    pod_timeout_seconds: int = 3600

class PreprocessSettings(BaseModel):
    worker: WorkerResourceSettings = WorkerResourceSettings()

class FeatureEngineerSettings(BaseModel):
    worker: WorkerResourceSettings = WorkerResourceSettings()

class ObjectStorageSettings(BaseModel):
    endpoint: str                              # required, e.g. http://minio:9000
    bucket: str                                # required
    access_key: str                            # required
    secret_key: str                            # required

class StorageSettings(BaseModel):
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    features: Path = Path("data/features")
    merged: Path = Path("data/merged")        # NEW
    # ... existing artifact paths

class Settings(BaseSettings):
    clearml: ClearmlSettings = ClearmlSettings()
    storage: StorageSettings = StorageSettings()
    object_storage: ObjectStorageSettings = ObjectStorageSettings()  # NEW
    k8s: K8sSettings = K8sSettings()                                 # NEW
    preprocess: PreprocessSettings = PreprocessSettings()            # NEW
    feature_engineer: FeatureEngineerSettings = FeatureEngineerSettings()  # NEW
```

`.env` example:
```bash
OBJECT_STORAGE__ENDPOINT=http://minio:9000
OBJECT_STORAGE__BUCKET=ml-pipeline
OBJECT_STORAGE__ACCESS_KEY=minioadmin
OBJECT_STORAGE__SECRET_KEY=minioadmin

K8S__NAMESPACE=ml-jobs
K8S__WORKER_IMAGE=registry.example.com/ml-project:latest
K8S__MAX_CONCURRENT_PODS=5
K8S__POD_TIMEOUT_SECONDS=3600

PREPROCESS__WORKER__CPU_REQUEST=2
PREPROCESS__WORKER__CPU_LIMIT=4
PREPROCESS__WORKER__MEMORY_REQUEST=4Gi
PREPROCESS__WORKER__MEMORY_LIMIT=8Gi

FEATURE_ENGINEER__WORKER__CPU_REQUEST=4
FEATURE_ENGINEER__WORKER__CPU_LIMIT=8
FEATURE_ENGINEER__WORKER__MEMORY_REQUEST=8Gi
FEATURE_ENGINEER__WORKER__MEMORY_LIMIT=16Gi
```

---

## Copier Template Changes

### New variable in `copier.yml`

```yaml
parallel_processing:
  type: bool
  default: false
  help: "Enable parallel file processing via Kubernetes pods"
```

### New files (`.jinja`, conditional on `parallel_processing`)

```
copier-python-template/
├── src/core/
│   ├── coordinator_step.py.jinja
│   ├── worker_step.py.jinja
│   └── merge_step.py.jinja
├── src/preprocess/
│   └── worker.py.jinja
├── src/features/
│   └── worker.py.jinja
├── src/merge/
│   ├── __init__.py
│   └── merge_pipeline_step.py.jinja
└── k8s/
    └── rbac.yaml.jinja
```

### Modified files (conditional jinja blocks added)

| File | Change |
|------|--------|
| `src/settings.py.jinja` | `+ K8sSettings`, `+ WorkerResourceSettings` per step, `+ StorageSettings.merged` |
| `src/common/pipeline_steps.py.jinja` | `+ MERGE` constant |
| `src/pipelines/pipeline.py.jinja` | New DAG: `SPLIT_DATASET` removed, `MERGE` added |
| `src/preprocess/preprocess_pipeline_step.py.jinja` | Inherits `BaseCoordinatorStep` when `parallel_processing=true` |
| `src/features/feature_engineer_pipeline_step.py.jinja` | Inherits `BaseCoordinatorStep` when `parallel_processing=true` |

### `k8s/rbac.yaml.jinja`

Generated only when `parallel_processing=true`. Contains `ServiceAccount`, `Role` (pods: create/get/list/watch + pods/log: get), and `RoleBinding` for `clearml-worker` in the configured namespace.

---

## Agent Checklist

Before modifying `BaseCoordinatorStep` or any coordinator step:
- [ ] Does `output_storage_path()` return a path from `StorageSettings`, not a hardcoded string?
- [ ] Does `_run_workers()` respect `SETTINGS.k8s.max_concurrent_pods`?
- [ ] Does a failed pod log to ClearML and continue (not raise immediately)?
- [ ] Does `start()` still accept `dataset_id: str` and return `str`?
- [ ] Does the coordinator read file list from ClearML Dataset metadata only (no download)?

Before modifying `BaseWorkerStep` or any worker:
- [ ] Does the worker read `INPUT_FILE_PATH`, `OUTPUT_PATH`, `STEP_PARAMS_JSON` from env?
- [ ] Does the worker have zero ClearML imports?
- [ ] Does `main()` call `cls().run()` as entry point?
