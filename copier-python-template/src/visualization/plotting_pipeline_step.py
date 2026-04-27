# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
from pathlib import Path
from typing import List, Union
import warnings

from clearml import Dataset
import pandas as pd
from tqdm import tqdm

from src.common.constants import GENERAL_EXTENSION
from src.common.exceptions import DatasetDownloadError
from src.common.features import GROUP_ID
from src.common.pipeline_steps import PLOTTING, PREPROCESS
from src.core import BasePipelineStep
from src.utilities.loaders import PickleLoader
from src.utilities.utils import split_dataframe
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
        self._download_preprocessed_dataset()
        processed = pd.DataFrame()
        for file_path in glob(str(self.settings.storage.processed_folder) + f"/*{GENERAL_EXTENSION}"):
            data = PickleLoader(path=Path(file_path)).load()
            processed = pd.concat([processed, data])
            del data
            gc.collect()
        return processed

    def _create_plot(self):
        groups = list(self.processed[GROUP_ID.name].unique().astype(int))
        for group in tqdm(groups, total=len(groups)):
            label = str(group)
            mask = self.processed[GROUP_ID.name] == group
            ldf = self.processed[mask].copy()
            splitted_ldf = split_dataframe(ldf)

            for i, small_ldf in enumerate(splitted_ldf):
                plotter = Plotter(
                    data=small_ldf,
                    filepath=Path(os.path.join(self._output_directory, f"{label}_part{i+1}")),
                    title=label,
                )
                _ = plotter.plot()

    def _process_data(self) -> None:
        self.prediction = self._get_data(self._input_directory)
        self.processed = self._get_processed_data()
        self.processed = pd.concat([self.processed, self.prediction], axis="columns")
        self._create_plot()












