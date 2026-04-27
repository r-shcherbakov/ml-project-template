# -*- coding: utf-8 -*-
"""ClearML PipelineController entry point: PREPROCESS → FEATURE_ENGINEER → TRAIN."""
from clearml import PipelineController

from src.common.pipeline_steps import PREPROCESS, FEATURE_ENGINEER, TRAIN
from src.features.feature_engineer_pipeline_step import FeatureEngineerPipelineStep, FeatureEngineerParams
from src.preprocess.preprocess_pipeline_step import PreprocessPipelineStep, PreprocessParams
from src.train.train_pipeline_step import TrainPipelineStep, TrainParams
from src.settings import SETTINGS


def run_preprocess_step() -> str:
    return PreprocessPipelineStep(
        params=PreprocessParams(skip_mark=False),
    ).start(dataset_id=None)


def run_feature_engineer_step(dataset_id: str) -> str:
    return FeatureEngineerPipelineStep(
        params=FeatureEngineerParams(),
    ).start(dataset_id=dataset_id)


def run_train_step(dataset_id: str) -> None:
    TrainPipelineStep(
        params=TrainParams(
            skip_cv=True,
            n_splits=4,
            train_final_model=True,
            binary_threshold=0.5,
        ),
    ).start(dataset_id=dataset_id)


if __name__ == '__main__':
    if SETTINGS.clearml.execute_remotely:
        assert SETTINGS.clearml.preprocess_queue, \
            "CLEARML__PREPROCESS_QUEUE must be set for remote execution"
        assert SETTINGS.clearml.feature_engineer_queue, \
            "CLEARML__FEATURE_ENGINEER_QUEUE must be set for remote execution"

    pipe = PipelineController(
        name=f'{SETTINGS.clearml.project} pipeline',
        project=SETTINGS.clearml.project,
        target_project=SETTINGS.clearml.project,
        add_pipeline_tags=False,
        auto_version_bump=True,
        add_run_number=False,
        packages="./requirements.txt",
    )

    pipe.add_function_step(
        name=PREPROCESS.name,
        task_type=PREPROCESS.task_type,
        function=run_preprocess_step,
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=FEATURE_ENGINEER.name,
        task_type=FEATURE_ENGINEER.task_type,
        parents=[PREPROCESS.name],
        function=run_feature_engineer_step,
        function_kwargs=dict(
            dataset_id='${preprocess.dataset_id}',
        ),
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=TRAIN.name,
        task_type=TRAIN.task_type,
        parents=[FEATURE_ENGINEER.name],
        function=run_train_step,
        function_kwargs=dict(
            dataset_id='${feature_engineer.dataset_id}',
        ),
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.set_default_execution_queue(SETTINGS.clearml.queue_name)
    if SETTINGS.clearml.execute_remotely:
        pipe.start(queue=SETTINGS.clearml.queue_name)
    else:
        pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline finished")
