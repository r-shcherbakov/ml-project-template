import os
from pathlib import Path
from typing import Tuple, TYPE_CHECKING

from clearml import PipelineController, Dataset
import pandas as pd

from src.common.exceptions import PipelineExecutionError
from src.common.pipeline_steps import (
    PRERUN,
    PREPROCESS,
    FEATURE_ENGINEER,
    SPLIT_DATASET,
    TRAIN,
)
from src.features import (
    FeatureEngineerPipelineStep,
    SplitDatasetPipelineStep,
)
from src.preprocess import PreprocessPipelineStep
from src.train import TrainPipelineStep
from src.settings import SETTINGS
from src.utilities.path_utils import is_empty_dir

if TYPE_CHECKING:
    from src.features.feature_engineer import FeatureEngineer


def run_prerun_step() -> str:
    if not is_empty_dir(SETTINGS.storage.raw_folder):
        try:
            # Upload raw data from remote storage
            pass
        except Exception:
            raise PipelineExecutionError("Raw data is not available")

        # Save local copy of raw data as ClearML Dataset at remote
        remote_dataset = Dataset.create(
            dataset_project=SETTINGS.clearml.project,
            dataset_name="raw data",
            dataset_tags=SETTINGS.clearml.tags,
        )
        remote_dataset.add_files(path=SETTINGS.storage.raw_folder)
        remote_dataset.finalize(auto_upload=True)

        return remote_dataset.id


def run_preprocess_step(dataset_id: str) -> str:
    return PreprocessPipelineStep().start(dataset_id=dataset_id)


def run_split_dataset_step(dataset_id: str) -> str:
    return SplitDatasetPipelineStep().start(dataset_id=dataset_id)


def run_feature_engineer_step(dataset_id: str) -> Tuple['FeatureEngineer', str]:
    return FeatureEngineerPipelineStep.start(dataset_id=dataset_id)


def run_train_step(dataset_id: str) -> None:
    return TrainPipelineStep.start(dataset_id=dataset_id)


if __name__ == '__main__':

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
        name=PRERUN.name,
        task_type=PRERUN.task_type,
        function=run_prerun_step,
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=PREPROCESS.name,
        task_type=PREPROCESS.task_type,
        parents=[PRERUN.name],
        function=run_preprocess_step,
        function_kwargs=dict(
            dataset_id='${prerun.dataset_id}'
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
        name=SPLIT_DATASET.name,
        task_type=SPLIT_DATASET.task_type,
        parents=[PREPROCESS.name],
        function=run_split_dataset_step,
        function_kwargs=dict(
            dataset_id='${preprocess.dataset_id}'
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
        name=FEATURE_ENGINEER.name,
        task_type=FEATURE_ENGINEER.task_type,
        parents=[SPLIT_DATASET.name],
        function=run_feature_engineer_step,
        function_kwargs=dict(
            dataset_id='${split_dataset.dataset_id}',
        ),
        function_return=['feature_engineer', 'dataset_id'],
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
        # Starting the pipeline (in the background)
        pipe.start(queue=SETTINGS.clearml.queue_name)
    else:
        # for debugging purposes use local jobs
        pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline finished")
