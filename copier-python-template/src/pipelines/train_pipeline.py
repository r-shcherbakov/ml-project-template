from clearml.automation import PipelineController

from settings import ExperimentSettings


settings = ExperimentSettings()

def post_execute_callback(a_pipeline: PipelineController, a_node: PipelineController.Node) -> None:
    print("Completed Task id={}".format(a_node.executed))
    
    
# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name=f"{settings.clearml.project} tasks pipeline", 
    project=settings.clearml.project, 
    version="0.0.1",
    add_pipeline_tags=False,
    retry_on_failure=3,
    auto_version_bump=True,
)

pipe.add_step(
    name="preprocess_task",
    base_task_project=settings.clearml.project,
    base_task_name="Pipeline task preprocess",
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name="feature_engineer_task",
    parents=["preprocess_task"],
    base_task_project=settings.clearml.project,
    base_task_name="Pipeline task feature engineer",
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name="split_dataset_task",
    parents=["feature_engineer_task"],
    base_task_project=settings.clearml.project,
    base_task_name="Pipeline task split dataset",
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name="train_task",
    parents=["split_dataset_task"],
    base_task_project=settings.clearml.project,
    base_task_name="Pipeline task train",
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
)

pipe.add_step(
    name="prediction_task",
    parents=["train_task"],
    base_task_project=settings.clearml.project,
    base_task_name="Pipeline task prediction",
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
)

pipe.set_default_execution_queue("default")
if settings.clearml.execute_remotely:
    # Starting the pipeline (in the background)
    pipe.start()
else:
    # for debugging purposes use local jobs
    pipe.start_locally(run_pipeline_steps_locally=True)

print("Pipeline successfully finished")