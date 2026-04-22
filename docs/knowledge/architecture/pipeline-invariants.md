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
