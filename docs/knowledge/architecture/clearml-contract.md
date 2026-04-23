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
