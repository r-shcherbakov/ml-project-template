# Knowledge Base Design

**Date**: 2026-04-22  
**Status**: Approved  
**Audience**: AI agents (Claude Code and equivalents)

## Problem

AI agents working in this repository lack structured context about:
- Architectural invariants they must not violate
- How the ClearML integration contract works
- How to correctly extend the template or a generated project

Without this context, agents make incorrect structural changes — breaking the dataset_id chain, misusing ClearML Task lifecycle, or adding pipeline steps that don't conform to the base class contract.

## Goal

A hierarchical knowledge base that:
1. Serves agents working on **the template itself** (modifying `copier-python-template/`, `copier.yml`)
2. Serves agents working in **projects generated from the template** (after `copier copy`)
3. Is navigable via `CLAUDE.md` index → topic file → specific invariant

## Chosen Approach

Domain-folder hierarchy under `docs/knowledge/`. Each file covers one architectural domain and follows a consistent structure agents can rely on.

## File Structure

```
docs/knowledge/
├── architecture/
│   ├── pipeline-invariants.md
│   ├── clearml-contract.md
│   ├── settings-model.md
│   └── feature-dataclass-contract.md
├── template/
│   ├── copier-variables.md
│   └── jinja-conventions.md
└── generated-project/
    ├── how-to-add-pipeline-step.md
    └── params-yaml-contract.md
```

`CLAUDE.md` in the repo root gains a `## Knowledge Base` section with one-line descriptions linking to each file.

## File Template (uniform across all files)

```markdown
# [Topic]

## Context
Why this is designed this way.

## Invariants
Numbered list of rules an agent must not violate.

## How It Works
Mechanism explanation.

## Agent Checklist
Bullet list: verify these before making changes.
```

## File Contents

### `architecture/pipeline-invariants.md`
- `dataset_id: str` is the **only** data transfer mechanism between steps
- Every `BasePipelineStep.start()` must return `str` (the output dataset_id)
- Steps must never read files from a sibling step's directory directly — always via ClearML Dataset download
- Pipeline step order is a DAG; no cycles, no skipping steps

### `architecture/clearml-contract.md`
- `deferred_init=True` in `Task.init()` — must not be removed (prevents premature task registration)
- `execute_remotely()` is called in `BasePipelineStep.__init__()`, not in `start()` — the step serializes before any data work
- `task.connect(params, name=...)` binds `params.yaml` sections to ClearML task parameters
- Dataset upload: `Dataset.create()` → `add_files()` → `finalize(auto_upload=True)` — all three calls are mandatory

### `architecture/settings-model.md`
- Full map of env variables → Pydantic fields (nested via `__` delimiter, e.g. `CLEARML__EXECUTE_REMOTELY`)
- Storage directories are created in `@field_validator`, not at runtime — adding new directories follows this pattern
- Minimum `.env` for local execution: no required fields (all have defaults); remote execution requires `CLEARML__EXECUTE_REMOTELY=true`

### `architecture/feature-dataclass-contract.md`
- `Feature` is a `frozen=True` dataclass — never mutate instances
- `MANDATORY_FEATURES` list in `src/common/features.py` is the single source of truth for clip/fillna/dtype configs
- Adding a feature: add to `MANDATORY_FEATURES` → `config.py` configs update automatically
- Required fields: `name`, `dtype`. All others optional.

### `template/copier-variables.md`
- Table: variable → type → default → which `.jinja` files use it
- Covers all variables defined in `copier.yml`

### `template/jinja-conventions.md`
- `_templates_suffix: .jinja` — only `.jinja` files are rendered; plain files are copied as-is
- `_subdirectory: copier-python-template` — template root is not the repo root
- `_skip_if_exists` — files listed here are never overwritten on `copier update`
- Adding a new templated file: place in `copier-python-template/`, add `.jinja` suffix, use `{{ variable }}` syntax

### `generated-project/how-to-add-pipeline-step.md`
5-step recipe:
1. Declare `PipelineStep` dataclass constant in `src/common/pipeline_steps.py`
2. Create step class inheriting `BasePipelineStep` in appropriate `src/<domain>/` module
3. Implement `start(dataset_id: str) -> str`
4. Add `add_function_step()` call in `src/pipelines/pipeline.py`
5. Add step parameters section in `src/params.yaml` (key = `PipelineStep.name`)

### `generated-project/params-yaml-contract.md`
- Top-level key in `params.yaml` must exactly match `PipelineStep.name`
- `BasePipelineStep._init_parameters()` reads the matching section automatically
- Missing key → `step_params = None` (no error, but no ClearML parameter binding)
- `common` key is reserved for cross-step parameters

## Branch Strategy

All knowledge base files are created on a dedicated branch `docs/knowledge-base`, then merged to `develop`.

## Success Criteria

An agent reading `CLAUDE.md` + one domain file can:
1. Identify which invariants apply to its task
2. Follow the correct pattern without reading source code
3. Not break the ClearML dataset chain or Task lifecycle
