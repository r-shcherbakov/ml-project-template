# Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a structured knowledge base of markdown files under `docs/knowledge/` that gives AI agents the architectural invariants, contracts, and recipes they need to work correctly in this repository and in projects generated from it.

**Architecture:** Hierarchical domain folders (`architecture/`, `template/`, `generated-project/`) each containing focused markdown files with a uniform four-section structure (Context → Invariants → How It Works → Agent Checklist). `CLAUDE.md` in the repo root serves as the navigable index.

**Tech Stack:** Markdown, Git (branch `docs/knowledge-base`), no code changes.

---

### Task 1: Create branch and directory structure

**Files:**
- Create: `docs/knowledge/architecture/` (directory)
- Create: `docs/knowledge/template/` (directory)
- Create: `docs/knowledge/generated-project/` (directory)

- [ ] **Step 1: Create the branch**

```bash
git checkout -b docs/knowledge-base
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p docs/knowledge/architecture
mkdir -p docs/knowledge/template
mkdir -p docs/knowledge/generated-project
```

- [ ] **Step 3: Verify directories exist**

```bash
find docs/knowledge -type d
```

Expected output:
```
docs/knowledge
docs/knowledge/architecture
docs/knowledge/template
docs/knowledge/generated-project
```

---

### Task 2: `architecture/pipeline-invariants.md`

**Files:**
- Create: `docs/knowledge/architecture/pipeline-invariants.md`

- [ ] **Step 1: Write the file**

Create `docs/knowledge/architecture/pipeline-invariants.md` with this exact content:

```markdown
# Pipeline Invariants

## Context

The ClearML pipeline passes data between steps exclusively via dataset IDs.
This design enables remote execution, step caching, and full reproducibility.
Breaking this contract causes silent data corruption or pipeline failures that
are difficult to debug because the error manifests far from its source.

## Invariants

1. `dataset_id: str` is the **only** data transfer mechanism between steps. Never pass DataFrames, file paths, or objects between steps directly.
2. Every `BasePipelineStep.start()` must accept `dataset_id: str` and return `str` (the output dataset_id).
3. Steps must never read files from a sibling step's local directory directly — always download via `self._download_input_dataset(dataset_id)`.
4. Pipeline step order is a DAG; no cycles. Steps are **optional** — any step can be omitted from `pipeline.py` entirely. If a step IS included, all steps listed in its `parents=[...]` must also be included and must complete successfully before it runs.
5. Each step uploads its output as a new ClearML Dataset that is a child of the input dataset (`parent_datasets=[dataset_id]`).

## How It Works

```
prerun        → uploads raw data as ClearML Dataset → returns dataset_id
preprocess    → downloads raw Dataset, processes, uploads processed Dataset → returns dataset_id
split_dataset → downloads processed Dataset, splits, uploads split Dataset → returns dataset_id
feature_engineer → downloads split Dataset, engineers features, uploads → returns dataset_id
train         → downloads features Dataset, trains model → (no dataset return required)
```

`PipelineController.add_function_step()` wires dataset_ids between steps via
template strings like `'${preprocess.dataset_id}'`. The `function_return` list
declares which return values are available to downstream steps.

## Agent Checklist

Before modifying or adding pipeline steps:
- [ ] Does `start()` accept `dataset_id: str` and return `str`?
- [ ] Is all inter-step data exchanged via ClearML Dataset, not local file paths?
- [ ] If adding a step, is it registered in `pipeline.py` with correct `parents=[...]`?
- [ ] Does `add_function_step()` include `function_return=['dataset_id']` if a downstream step uses it?
- [ ] Is the upload sequence `Dataset.create()` → `add_files()` → `finalize(auto_upload=True)`?
```

- [ ] **Step 2: Verify file contains all required sections**

```bash
grep -c "^## " docs/knowledge/architecture/pipeline-invariants.md
```

Expected output: `4` (Context, Invariants, How It Works, Agent Checklist)

- [ ] **Step 3: Commit**

```bash
git add docs/knowledge/architecture/pipeline-invariants.md
git commit -m "docs: add pipeline invariants knowledge base entry"
```

---

### Task 3: `architecture/clearml-contract.md`

**Files:**
- Create: `docs/knowledge/architecture/clearml-contract.md`

- [ ] **Step 1: Write the file**

Create `docs/knowledge/architecture/clearml-contract.md` with this exact content:

```markdown
# ClearML Contract

## Context

ClearML handles experiment tracking, dataset versioning, and remote execution.
The integration is established in `BasePipelineStep.__init__()` before any data
processing begins. Misusing the Task lifecycle causes tasks to register
incorrectly, fail to serialize for remote execution, or lose parameter bindings.

## Invariants

1. `deferred_init=True` must not be removed from `Task.init()`. It prevents premature task registration before all parameters are connected via `task.connect()`.
2. `task.execute_remotely(queue_name=...)` is called in `__init__()`, not in `start()`. The entire step serializes at this point — any code after this line runs on the remote worker.
3. `task.connect(params, name=...)` must be called before `start()` executes. This is handled by `_init_parameters()` in `__init__()` — do not move it.
4. Dataset upload sequence is mandatory and must not be reordered: `Dataset.create()` → `add_files()` → `finalize(auto_upload=True)`.
5. Never call `task.close()` manually — ClearML manages task lifecycle.
6. `reuse_last_task_id=False` must remain set — each pipeline run creates a new task, not reuse a previous one.

## How It Works

```python
# BasePipelineStep.__init__() — order matters
self.task = Task.init(
    project_name=self.settings.clearml.project,
    task_name=f'{self.pipeline_step.name} task',
    task_type=self.pipeline_step.task_type,
    tags=self.settings.clearml.tags,
    deferred_init=True,          # ← prevents premature registration
    reuse_last_task_id=False,    # ← always a fresh task
)
if self.settings.clearml.execute_remotely:
    self.task.execute_remotely(queue_name=self.settings.clearml.queue_name)
    # ← everything after this runs on the remote worker

# _init_parameters() — called immediately after in __init__()
self.task.connect(self.common_params, name="common")
self.task.connect(self.step_params, name=self.pipeline_step.name)
```

Remote execution flow:
1. Local machine calls `__init__()`, which enqueues the task
2. Remote worker picks up the task and re-runs the step from `start()`
3. All parameters are already bound via `task.connect()` and visible in ClearML UI

## Agent Checklist

Before modifying `BasePipelineStep` or any step's `__init__()`:
- [ ] Is `deferred_init=True` still present in `Task.init()`?
- [ ] Is `reuse_last_task_id=False` still present?
- [ ] Is `execute_remotely()` called before any data access?
- [ ] Are all `task.connect()` calls made before `start()` runs?
- [ ] Does dataset upload follow `Dataset.create()` → `add_files()` → `finalize(auto_upload=True)` order?
```

- [ ] **Step 2: Verify file contains all required sections**

```bash
grep -c "^## " docs/knowledge/architecture/clearml-contract.md
```

Expected output: `4`

- [ ] **Step 3: Commit**

```bash
git add docs/knowledge/architecture/clearml-contract.md
git commit -m "docs: add ClearML contract knowledge base entry"
```

---

### Task 4: `architecture/settings-model.md`

**Files:**
- Create: `docs/knowledge/architecture/settings-model.md`

- [ ] **Step 1: Write the file**

Create `docs/knowledge/architecture/settings-model.md` with this exact content:

```markdown
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
```

- [ ] **Step 2: Verify**

```bash
grep -c "^## " docs/knowledge/architecture/settings-model.md
```

Expected output: `4`

- [ ] **Step 3: Commit**

```bash
git add docs/knowledge/architecture/settings-model.md
git commit -m "docs: add settings model knowledge base entry"
```

---

### Task 5: `architecture/feature-dataclass-contract.md`

**Files:**
- Create: `docs/knowledge/architecture/feature-dataclass-contract.md`

- [ ] **Step 1: Write the file**

Create `docs/knowledge/architecture/feature-dataclass-contract.md` with this exact content:

```markdown
# Feature Dataclass Contract

## Context

Features are defined as frozen dataclasses in `src/common/features.py`.
`MANDATORY_FEATURES` is the single source of truth — any change here
automatically propagates to clip configs, fillna configs, and dtype configs
in `src/common/config.py`. This eliminates manual synchronization across files.

## Invariants

1. `Feature` is `frozen=True` — never mutate instances. Never use `dataclasses.replace()` to work around immutability.
2. `MANDATORY_FEATURES` in `features.py` is the only place to register features that need clip/fillna/dtype enforcement. Do not add them anywhere else.
3. Required fields for any `Feature` used in `MANDATORY_FEATURES`: `name` (str) and `dtype` (str). All other fields (`lower`, `upper`, `fillna_value`, `fillna_method`, `fillna_limit`) are optional.
4. `IGNORED_FEATURES` lists features present in the data but excluded from processing (e.g., `GROUP_ID`, grouping columns, ID columns). Add such columns here, not to `MANDATORY_FEATURES`.
5. Never edit `src/common/config.py` directly — `FEATYPE_TYPES`, `CLIP_CONFIG`, and `FILLNA_CONFIG` are derived from `MANDATORY_FEATURES` at module load time.

## How It Works

```python
# features.py — single source of truth
MANDATORY_FEATURES = [FEATURE_1, FEATURE_2, FEATURE_3]

# config.py — auto-derived at import time, never edit manually
FEATYPE_TYPES = {f.name: f.dtype for f in MANDATORY_FEATURES}

CLIP_CONFIG = {
    f.name: {"lower": f.lower, "upper": f.upper}
    for f in MANDATORY_FEATURES
}

FILLNA_CONFIG = {
    f.name: {"value": f.fillna_value, "method": f.fillna_method, "limit": f.fillna_limit}
    for f in MANDATORY_FEATURES
}
```

To add a new feature that needs processing:
```python
# 1. Define in features.py
MY_FEATURE = Feature(
    name="my_feature",
    dtype="float32",
    lower=0.0,
    upper=100.0,
    fillna_value=0.0,
)

# 2. Add to MANDATORY_FEATURES (config.py updates automatically)
MANDATORY_FEATURES = [FEATURE_1, FEATURE_2, FEATURE_3, MY_FEATURE]
```

## Agent Checklist

Before adding or modifying a feature:
- [ ] Is the `Feature` instance defined in `src/common/features.py`?
- [ ] Is it added to `MANDATORY_FEATURES` (if it needs clip/fillna/dtype enforcement)?
- [ ] Does it have both `name` and `dtype` set?
- [ ] Is `src/common/config.py` left untouched (configs regenerate automatically)?
- [ ] If it's a grouping/ID column, is it in `IGNORED_FEATURES` instead?
```

- [ ] **Step 2: Verify**

```bash
grep -c "^## " docs/knowledge/architecture/feature-dataclass-contract.md
```

Expected output: `4`

- [ ] **Step 3: Commit**

```bash
git add docs/knowledge/architecture/feature-dataclass-contract.md
git commit -m "docs: add feature dataclass contract knowledge base entry"
```

---

### Task 6: `template/copier-variables.md`

**Files:**
- Create: `docs/knowledge/template/copier-variables.md`

- [ ] **Step 1: Write the file**

Create `docs/knowledge/template/copier-variables.md` with this exact content:

```markdown
# Copier Template Variables

## Context

All template variables are declared in `copier.yml`. When a user runs
`copier copy`, Copier prompts for each variable and makes them available
as Jinja2 variables in all `.jinja` files under `copier-python-template/`.
Never hardcode values that should vary per project — use variables.

## Invariants

1. All user-facing variables must be declared in `copier.yml` with `type` and `help`. Provide `default` where appropriate.
2. `package_name_py` must always be a valid Python identifier — the default transformation (`lower|replace('-','_')|replace(' ','_')`) handles this automatically.
3. `doc_dir` controls the documentation directory name — the `{{doc_dir}}` folder in the template becomes its value (default: `docs/`).
4. `docker` is the only boolean variable; use `{% if docker %}...{% endif %}` for optional sections.
5. When adding a new variable, update this file's Variable Reference table and all `.jinja` files that use it.

## Variable Reference

| Variable | Type | Default | Used In |
|---|---|---|---|
| `project_name` | str | *(required)* | `settings.py.jinja` (ClearmlSettings.project), `README.md.jinja` |
| `package_name_py` | str | slugified `project_name` | `pyproject.toml.jinja`, `tests/__init__.py.jinja`, `tests/test_zz_*.py.jinja`, `{{ _copier_conf.answers_file }}.jinja` |
| `project_description` | str | *(required)* | `pyproject.toml.jinja`, `README.md.jinja` |
| `python_version` | str (choice) | *(required)* | `pyproject.toml.jinja` (`[tool.poetry.dependencies]`) |
| `author_name` | str | `Name.Surname` | `pyproject.toml.jinja`, `AUTHORS.tmpl` |
| `author_email` | str | `{author_name}@mail.com` | `pyproject.toml.jinja` |
| `repository_provider` | str | *(required)* | `pyproject.toml.jinja`, `{{doc_dir}}/DEVELOPER_GUIDE.md.jinja` |
| `project_namespace` | str | *(required)* | `pyproject.toml.jinja`, `{{doc_dir}}/DEVELOPER_GUIDE.md.jinja` |
| `copyright_date` | str | current year | used in license/copyright blocks |
| `doc_dir` | str | `docs/` | controls `{{doc_dir}}/` directory name; `pyproject.toml.jinja` |
| `docker` | bool | `false` | conditional sections in `pyproject.toml.jinja` |

## How It Works

Copier reads `copier.yml`, prompts the user for each variable without a default,
then renders all `.jinja` files substituting `{{ variable_name }}` with the
provided value. The rendered files are written to the destination directory
without the `.jinja` suffix.

## Agent Checklist

Before adding a new template variable:
- [ ] Is it declared in `copier.yml` with `type`, `help`, and `default` (where applicable)?
- [ ] Is it added to the Variable Reference table in this file?
- [ ] Are all `.jinja` files that use it listed in the table?
- [ ] If boolean, is `{% if variable %}` used for gating optional content?
```

- [ ] **Step 2: Verify**

```bash
grep -c "^## " docs/knowledge/template/copier-variables.md
```

Expected output: `4`

- [ ] **Step 3: Commit**

```bash
git add docs/knowledge/template/copier-variables.md
git commit -m "docs: add copier variables knowledge base entry"
```

---

### Task 7: `template/jinja-conventions.md`

**Files:**
- Create: `docs/knowledge/template/jinja-conventions.md`

- [ ] **Step 1: Write the file**

Create `docs/knowledge/template/jinja-conventions.md` with this exact content:

```markdown
# Jinja Template Conventions

## Context

Copier uses Jinja2 to render template files. Only files with the `.jinja` suffix
are rendered — plain files are copied verbatim. The template root is
`copier-python-template/` (set via `_subdirectory` in `copier.yml`), not the
repo root. This means all paths in `copier.yml` config keys are relative to
`copier-python-template/`.

## Invariants

1. Only `.jinja` files are rendered. To template a file, rename it to `<name>.jinja`. The `.jinja` suffix is stripped in the output (e.g., `settings.py.jinja` → `settings.py`).
2. Template root is `copier-python-template/` — the repo root is never the template source.
3. Files listed in `_skip_if_exists` in `copier.yml` are **never overwritten** on `copier update`. Use this for files users are expected to customize (`pyproject.toml`, `__init__.py`, `README.md`).
4. Standard Jinja2 delimiters apply: `{{ variable }}` for substitution, `{% if/for/block %}` for logic, `{# comment #}` for comments.
5. Directory names can also be templated: `{{doc_dir}}/` becomes the value of the `doc_dir` variable (e.g., `docs/`). Use this pattern for optional or configurable directories.
6. `_tasks` in `copier.yml` run shell commands after rendering. Keep tasks idempotent (guarded by `git log &>/dev/null ||`).

## How to Add a New Templated File

1. Create the file in `copier-python-template/` at the desired output path.
2. Add `.jinja` suffix: `src/myconfig.py` → `src/myconfig.py.jinja`.
3. Use `{{ variable_name }}` for substitutions from variables declared in `copier.yml`.
4. Use `{% if condition %}...{% endif %}` for sections gated on boolean variables.
5. If users should customize this file after generation (not have it overwritten on `copier update`), add its path to `_skip_if_exists` in `copier.yml`.

## How to Add a New Plain (Non-Templated) File

1. Create the file in `copier-python-template/` without `.jinja` suffix.
2. Do not add it to any special config — Copier copies it verbatim.

## Agent Checklist

Before adding or modifying a template file:
- [ ] Does the file have `.jinja` suffix if it needs variable substitution?
- [ ] Are only variables declared in `copier.yml` used inside `{{ }}`?
- [ ] If user-customizable, is its path in `_skip_if_exists` in `copier.yml`?
- [ ] If adding a variable-named directory, does the folder use `{{variable}}` syntax?
- [ ] If adding a `_tasks` entry, is it guarded against re-running on existing repos?
```

- [ ] **Step 2: Verify**

```bash
grep -c "^## " docs/knowledge/template/jinja-conventions.md
```

Expected output: `5` (Context, Invariants, How to Add Templated, How to Add Plain, Agent Checklist)

- [ ] **Step 3: Commit**

```bash
git add docs/knowledge/template/jinja-conventions.md
git commit -m "docs: add jinja conventions knowledge base entry"
```

---

### Task 8: `generated-project/how-to-add-pipeline-step.md`

**Files:**
- Create: `docs/knowledge/generated-project/how-to-add-pipeline-step.md`

- [ ] **Step 1: Write the file**

Create `docs/knowledge/generated-project/how-to-add-pipeline-step.md` with this exact content:

```markdown
# How to Add a Pipeline Step

## Context

Pipeline steps are the fundamental unit of work in generated projects. Each step
downloads its input dataset from ClearML, processes data, and uploads an output
dataset. A step must be declared as a `PipelineStep` constant, implemented as a
`BasePipelineStep` subclass, registered in `pipeline.py`, and have its parameters
in `params.yaml`.

## Invariants

1. `PipelineStep.name` must exactly match the top-level key in `params.yaml` (case-sensitive).
2. `start(dataset_id: str) -> str` — this signature is mandatory. The return value is the output dataset_id.
3. Output directories must be declared as `@computed_field` in `StorageSettings`, not created ad-hoc in step code.
4. A step is optional — omitting it from `pipeline.py` is valid. Including it requires all its `parents` to be included too.

## How It Works: 5-Step Recipe

### Step 1 — Declare the `PipelineStep` constant

In `src/common/pipeline_steps.py`:

```python
from src.settings import StorageSettings, ArtifactsSettings
from clearml import TaskTypes

storage_settings = StorageSettings()  # already at top of file

MY_STEP = PipelineStep(
    name="my_step",                               # must match params.yaml key exactly
    task_type=TaskTypes.data_processing.name,     # or training, service, optimizer
    input_directory=storage_settings.processed_folder,
    output_directory=storage_settings.my_step_folder,  # add this field to StorageSettings first
)
```

### Step 2 — Add output directory to `StorageSettings`

In `src/settings.py` (or `settings.py.jinja` in template), inside `StorageSettings`:

```python
@computed_field(description="Path to my step output data")
def my_step_folder(self) -> Path:
    directory = Path(os.path.join(self.root_folder, "my_step"))
    directory.mkdir(exist_ok=True, parents=True)
    return directory
```

### Step 3 — Implement the step class

Create `src/my_step/my_step_pipeline_step.py`:

```python
from src.common.pipeline_steps import MY_STEP
from src.core import BasePipelineStep
from src.utilities.path_utils import is_empty_dir


class MyStepPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(MY_STEP)

    def start(self, dataset_id: str) -> str:
        if is_empty_dir(self._input_directory):
            self._download_input_dataset(dataset_id=dataset_id)

        # Process files in self._input_directory
        # Save results with self._save_locally_data(path, dataframe)

        return self._upload_output_dataset(parent_datasets=[dataset_id])
```

Create `src/my_step/__init__.py`:
```python
from src.my_step.my_step_pipeline_step import MyStepPipelineStep
```

### Step 4 — Register in `pipeline.py`

In `src/pipelines/pipeline.py`:

```python
from src.common.pipeline_steps import MY_STEP
from src.my_step import MyStepPipelineStep


def run_my_step(dataset_id: str) -> str:
    return MyStepPipelineStep().start(dataset_id=dataset_id)


# Inside the PipelineController block:
pipe.add_function_step(
    name=MY_STEP.name,
    task_type=MY_STEP.task_type,
    parents=[PREVIOUS_STEP.name],
    function=run_my_step,
    function_kwargs=dict(dataset_id='${previous_step.dataset_id}'),
    function_return=['dataset_id'],
    cache_executed_step=True,
    continue_behaviour=dict(continue_on_fail=False, continue_on_abort=False),
    time_limit=SETTINGS.clearml.time_limit,
)
```

### Step 5 — Add parameters to `params.yaml`

In `src/params.yaml`, add a section whose key exactly matches `MY_STEP.name`:

```yaml
my_step:
  some_threshold: 0.5
  skip_validation: False
```

Access in step code via `self.step_params.get("some_threshold", 0.5)`.

## Agent Checklist

Before finalizing a new pipeline step:
- [ ] `PipelineStep.name` exactly matches the `params.yaml` top-level key
- [ ] `start()` signature is `(self, dataset_id: str) -> str`
- [ ] Output directory is a `@computed_field` in `StorageSettings`
- [ ] Step is registered in `pipeline.py` with correct `parents=[...]`
- [ ] `function_return=['dataset_id']` is present if a downstream step uses this step's output
- [ ] `src/my_step/__init__.py` exports the step class
```

- [ ] **Step 2: Verify**

```bash
grep -c "^## \|^### " docs/knowledge/generated-project/how-to-add-pipeline-step.md
```

Expected output: `9` or more (Context, Invariants, How It Works, Steps 1-5, Agent Checklist)

- [ ] **Step 3: Commit**

```bash
git add docs/knowledge/generated-project/how-to-add-pipeline-step.md
git commit -m "docs: add how-to-add-pipeline-step knowledge base entry"
```

---

### Task 9: `generated-project/params-yaml-contract.md`

**Files:**
- Create: `docs/knowledge/generated-project/params-yaml-contract.md`

- [ ] **Step 1: Write the file**

Create `docs/knowledge/generated-project/params-yaml-contract.md` with this exact content:

```markdown
# params.yaml Contract

## Context

`src/params.yaml` is the single file for all pipeline step parameters.
`BasePipelineStep._init_parameters()` reads it automatically on step
initialization and connects the matching section to the ClearML task —
making parameters visible and editable in the ClearML UI without code changes.
Parameters can then be overridden in ClearML UI for remote runs.

## Invariants

1. The top-level key in `params.yaml` must **exactly** match `PipelineStep.name` (case-sensitive, underscores preserved).
2. `common` is a reserved top-level key — its contents are connected to every task as the "common" parameter group. Use it only for truly cross-step parameters.
3. A missing step key results in `self.step_params = None` — no error is raised, but the step has no ClearML parameter binding.
4. All parameter values must be YAML-serializable primitives: `str`, `int`, `float`, `bool`, `list`, `dict`.
5. Never read `params.yaml` directly in step code — always use `self.step_params` (already loaded in `_init_parameters()`).

## How It Works

```yaml
# src/params.yaml structure
common:              # ← connected to ALL tasks as "common" parameter group
  mandatory_features:
    - target
  forbidden_features:
    - GROUP_ID

preprocess:          # ← PipelineStep.name = "preprocess" → self.step_params
  skip_mark: False   #    access: self.step_params.get("skip_mark", True)

feature_engineer:    # ← PipelineStep.name = "feature_engineer"
  some_int_parameter: 1

train:
  skip_cv: True
  n_splits: 4
  train_final_model: True
```

In `BasePipelineStep._init_parameters()`:
```python
with open(self.settings.params_path) as file:
    params = yaml.load(file, Loader=yaml.Loader)
    self.common_params = params.get('common', None)
    self.step_params = params.get(self.pipeline_step.name, None)  # key = PipelineStep.name

if self.common_params:
    self.task.connect(self.common_params, name="common")
if self.step_params:
    self.task.connect(self.step_params, name=self.pipeline_step.name)
```

## Agent Checklist

Before adding or modifying parameters:
- [ ] Is the top-level key identical to `PipelineStep.name`?
- [ ] Is `common` used only for parameters needed by multiple steps?
- [ ] Are all values YAML-serializable primitives?
- [ ] Is `self.step_params` used in step code (not direct `yaml.load()` calls)?
- [ ] If a step has no parameters, is omitting its key acceptable (results in `self.step_params = None`)?
```

- [ ] **Step 2: Verify**

```bash
grep -c "^## " docs/knowledge/generated-project/params-yaml-contract.md
```

Expected output: `4`

- [ ] **Step 3: Commit**

```bash
git add docs/knowledge/generated-project/params-yaml-contract.md
git commit -m "docs: add params.yaml contract knowledge base entry"
```

---

### Task 10: Update `CLAUDE.md` with Knowledge Base index

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Append the Knowledge Base section to `CLAUDE.md`**

Add this section at the end of `CLAUDE.md`:

```markdown
## Knowledge Base

Structured context for AI agents. Read `CLAUDE.md` first, then open the relevant domain file before making changes.

### Architecture (invariants that must not be violated)
- [`docs/knowledge/architecture/pipeline-invariants.md`](docs/knowledge/architecture/pipeline-invariants.md) — dataset_id chain, DAG ordering, inter-step data transfer rules
- [`docs/knowledge/architecture/clearml-contract.md`](docs/knowledge/architecture/clearml-contract.md) — Task lifecycle, deferred_init, execute_remotely, Dataset upload sequence
- [`docs/knowledge/architecture/settings-model.md`](docs/knowledge/architecture/settings-model.md) — Pydantic settings, env var format, directory creation pattern
- [`docs/knowledge/architecture/feature-dataclass-contract.md`](docs/knowledge/architecture/feature-dataclass-contract.md) — Feature frozen dataclass, MANDATORY_FEATURES as single source of truth

### Template Development (modifying `copier-python-template/` or `copier.yml`)
- [`docs/knowledge/template/copier-variables.md`](docs/knowledge/template/copier-variables.md) — all variables, types, defaults, which .jinja files use each
- [`docs/knowledge/template/jinja-conventions.md`](docs/knowledge/template/jinja-conventions.md) — .jinja suffix rules, _subdirectory, _skip_if_exists, adding new files

### Generated Project Recipes (working in a project created from this template)
- [`docs/knowledge/generated-project/how-to-add-pipeline-step.md`](docs/knowledge/generated-project/how-to-add-pipeline-step.md) — 5-step recipe for adding a new pipeline step
- [`docs/knowledge/generated-project/params-yaml-contract.md`](docs/knowledge/generated-project/params-yaml-contract.md) — params.yaml structure, key naming, step_params access
```

- [ ] **Step 2: Verify the section was added**

```bash
grep -c "knowledge" CLAUDE.md
```

Expected output: `9` or more

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add knowledge base index to CLAUDE.md"
```

---

### Task 11: Final verification

- [ ] **Step 1: Verify all files exist**

```bash
find docs/knowledge -name "*.md" | sort
```

Expected output:
```
docs/knowledge/architecture/clearml-contract.md
docs/knowledge/architecture/feature-dataclass-contract.md
docs/knowledge/architecture/pipeline-invariants.md
docs/knowledge/architecture/settings-model.md
docs/knowledge/generated-project/how-to-add-pipeline-step.md
docs/knowledge/generated-project/params-yaml-contract.md
docs/knowledge/template/copier-variables.md
docs/knowledge/template/jinja-conventions.md
```

- [ ] **Step 2: Verify all files have the four required sections**

```bash
for f in docs/knowledge/**/*.md; do
  count=$(grep -c "^## " "$f")
  echo "$count sections: $f"
done
```

Expected: each file reports 4 or more sections.

- [ ] **Step 3: Verify CLAUDE.md links are present**

```bash
grep "docs/knowledge" CLAUDE.md | wc -l
```

Expected output: `8` (one line per knowledge file)

- [ ] **Step 4: Check git log on this branch**

```bash
git log --oneline docs/knowledge-base..HEAD 2>/dev/null || git log --oneline -15
```

Expected: 10+ commits, one per task.
