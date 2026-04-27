# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
import logging
from pathlib import Path
from typing import Dict, List, Union
import warnings

from clearml import Dataset, Task
import pandas as pd
from tqdm import tqdm

from src.common.constants import GENERAL_EXTENSION
from src.common.exceptions import (
    DatasetDownloadError,
    PipelineExecutionError,
)
from src.common.features import GROUP_ID
from src.common.pipeline_steps import (
    PLOTTING,
    PREPROCESS,
)
from src.core import BasePipelineStep
from src.utilities.loaders import PickleLoader
from src.utilities.utils import (
    invert_dict,
    split_dataframe,
)
from src.utilities.path_utils import get_last_modified, is_empty_dir

warnings.simplefilter(action="ignore", category=FutureWarning)


class PlottingPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(PLOTTING)

    @property
    def _input_files(self) -> List[Path]:
        return []

    def _get_data(self, path: Union[Path, str]) -> pd.DataFrame:
        file_path = get_last_modified(path=path, suffixes=GENERAL_EXTENSION)
        data = PickleLoader(path=file_path).load()
        return data

    def _upload_artifacts(self) -> None:
        pass

    def _get_groups_mapping(self) -> Dict[str, int]:
        artifacts_task = Task.get_task(
            project_name=self.settings.clearml.project,
            task_name=f'{PREPROCESS.name} task',
            task_filter={'status': ['completed']},
        )
        groups_mapping = artifacts_task.artifacts['groups_mapping'].get()
        return groups_mapping

    def _download_preprocessed_dataset(self) -> None:
        if is_empty_dir(self.settings.storage.processed_folder):
            try:
                self.remote_dataset = Dataset.get(
                    dataset_project=self.settings.clearml.project,
                    dataset_name=f"{self.settings.clearml.project} {PREPROCESS.name} output dataset",

                )
            except ValueError:
                raise DatasetDownloadError

            _ = Path(
                    self.remote_dataset.get_mutable_local_copy(
                    self.settings.storage.processed_folder,
                    )
                )

    def _get_processed_data(self) -> pd.DataFrame:
        groups = list(self.prediction[GROUP_ID.name].unique().astype(int))
        self.groups_mapping = invert_dict(self._get_groups_mapping())
        self._download_preprocessed_dataset()
        processed = pd.DataFrame()
        for group in groups:
            file_name = self.groups_mapping.get(group)
            if file_name:
                preprocessed_filepath = Path(
                    os.path.join(
                        self.settings.storage.processed_folder, f"{file_name}{GENERAL_EXTENSION}"
                    )
                )
                data = PickleLoader(path=preprocessed_filepath).load()
                processed = pd.concat([processed, data])

                del data
                gc.collect()
            else:
                raise PipelineExecutionError

        return processed

    def _create_plot(self):
        groups = list(self.processed[GROUP_ID.name].unique().astype(int))
        for group in tqdm(groups, total=len(groups)):
            file_name = self.groups_mapping.get(group)
            mask = self.processed[GROUP_ID.name] == group
            ldf = self.processed[mask].copy()
            splitted_ldf = split_dataframe(ldf)

            plotter = Plotter()
            for i, small_ldf in enumerate(splitted_ldf):
                plotter = Plotter(
                    data=small_ldf,
                    filepath=Path(os.path.join(self._output_directory, f"{file_name}_part{i+1}")),
                    title=file_name,
                )
                _ = plotter.plot()

    def _process_data(self) -> None:
        self.prediction = self._get_data(self._input_directory)
        self.processed = self._get_processed_data()
        self.processed = pd.concat([self.processed, self.prediction], axis="columns")
        self._create_plot()












