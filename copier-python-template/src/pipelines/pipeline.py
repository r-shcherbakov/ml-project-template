from clearml.automation import PipelineController

from common.enums import PipelineSteps
from settings import Settings


settings = Settings()

def post_execute_callback(a_pipeline: PipelineController, a_node: PipelineController.Node) -> None:
    print('Completed Task id={}'.format(a_node.executed))
    
    
# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name=f'{settings.clearml.project} tasks pipeline', 
    project=settings.clearml.project, 
    version='0.0.1',
    add_pipeline_tags=False,
    retry_on_failure=3,
    auto_version_bump=True,
)

pipe.add_step(
    name=f'{PipelineSteps.preprocess.name} step',
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.preprocess.name} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name=f'{PipelineSteps.feature_engineer.name} step',
    parents=[f'{PipelineSteps.preprocess.name} step'],
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.feature_engineer.name} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name=f'{PipelineSteps.split_dataset.name} step',
    parents=[f'{PipelineSteps.feature_engineer.name} step'],
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.split_dataset.name} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name=f'{PipelineSteps.train.name} step',
    parents=[f'{PipelineSteps.split_dataset.name} step'],
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.train.name} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
)

pipe.add_step(
    name=f'{PipelineSteps.plotting.name} step',
    parents=[f'{PipelineSteps.train.name} step'],
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.plotting.name} task',
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