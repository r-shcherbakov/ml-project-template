# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A [Copier](https://copier.readthedocs.io/en/stable/) template that generates ClearML-tracked ML Python projects. The `copier-python-template/` directory is the template source — its files are Jinja2-rendered when a user runs `copier copy`. Files with `.jinja` suffix are templates; plain files are copied as-is (unless listed in `_skip_if_exists` in `copier.yml`).

## Commands

### Template development (this repo)
```bash
poetry install          # install dev dependencies (copier, pytest, pyright, yamllint)
pytest                  # run template smoke tests
yamllint copier.yml     # lint the template config
```

### Generated project (inside a project created from this template)
```bash
poetry install                                        # install dependencies
poetry run pytest                                     # run all tests
poetry run pytest tests/test_specific.py::test_name  # run a single test
python -m src.pipelines.pipeline                      # run full ClearML pipeline
```

## Architecture of Generated Projects

### Pipeline execution flow

```
prerun → preprocess → split_dataset → feature_engineer → train → (plotting, postrun optional)
```

Each step is a `BasePipelineStep` subclass. Steps are wired together via `ClearMLPipelineController` in `src/pipelines/pipeline.py`. Each step:
1. Downloads its input as a ClearML `Dataset` (by `dataset_id`)
2. Processes data locally
3. Uploads output as a new ClearML `Dataset`
4. Returns the new `dataset_id` to the next step

### Key abstractions (`src/core/`)

- **`BasePipelineStep`** — all pipeline steps inherit from this. Handles `Task.init()`, parameter loading from `params.yaml`, dataset up/download, and ClearML logging. Subclasses implement `start(dataset_id) -> str`.
- **`BaseTransformer`** — sklearn `BaseEstimator + TransformerMixin` base for all data transformers. Subclasses implement `transform()`.
- **`BaseLoader`** — abstract file loader. Subclasses implement `load()`.
- **`Metric`** — abstract metric accumulator. Results are stored across calls; `get_results()` returns the aggregated value.

### Configuration (`src/settings.py`)

Pydantic `BaseSettings` loaded from `.env`. Nested settings use `__` as delimiter (e.g. `CLEARML__EXECUTE_REMOTELY=true`, `CLEARML__QUEUE_NAME=gpu`). Key sections:
- `StorageSettings` — local data paths (`data/raw`, `data/processed`, etc.), auto-created on init
- `ArtifactsSettings` — output paths (`artifacts/models`, `artifacts/plots`, etc.), auto-created on init
- `ClearmlSettings` — project name, queue, remote execution toggle
- `params_path` — defaults to `src/params.yaml`

### Feature and step registration

- **Features** are declared as `Feature` dataclasses in `src/common/features.py`. `MANDATORY_FEATURES` drives the clip/fillna/dtype configs built in `src/common/config.py`.
- **Pipeline steps** are declared as frozen `PipelineStep` dataclasses in `src/common/pipeline_steps.py` with their ClearML `task_type` and I/O directories. The constants (`PRERUN`, `PREPROCESS`, etc.) are imported directly wherever needed — no registry.

### Parameters (`src/params.yaml`)

YAML file with one top-level key per pipeline step name (matching `PipelineStep.name`). `BasePipelineStep._init_parameters()` reads this and connects the step's section to the ClearML task automatically.

## Template Variables

Key variables defined in `copier.yml` that appear in `.jinja` files:
- `project_name`, `package_name_py` — project and Python package names
- `python_version` — one of `^3.8` / `^3.9` / `^3.10` / `^3.11`
- `author_name`, `author_email`, `repository_provider`, `project_namespace`
- `doc_dir` — documentation directory (default `docs/`)
- `docker` — bool, whether to include Docker config

## ClearML Remote Execution

Set `CLEARML__EXECUTE_REMOTELY=true` in `.env` to enqueue steps for remote execution. Each `BasePipelineStep.__init__` calls `task.execute_remotely(queue_name=...)` when this flag is set, so the step serializes and runs in the queue rather than locally.

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
