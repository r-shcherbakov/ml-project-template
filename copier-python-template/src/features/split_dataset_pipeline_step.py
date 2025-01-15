# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional
import warnings

import pandas as pd
from tqdm import tqdm

from src.common.constants import GENERAL_EXTENSION
from src.common.exceptions import PipelineExecutionError
from src.common.features import GROUP_ID
from src.common.pipeline_steps import SPLIT_DATASET
from src.core import BasePipelineStep
from src.utilities.loaders import PickleLoader
from src.utilities.path_utils import is_empty_dir

warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitDatasetPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(SPLIT_DATASET)

    @property
    def _input_files(self) -> List[Path]:
        self._check_input_directory()
        input_directory = self._input_directory
        file_type = f"/*{GENERAL_EXTENSION}"
        input_filepath_files = [
            Path(file_path) for file_path in glob(str(input_directory) + file_type)
        ]
        return input_filepath_files

    def _set_test_objects(self) -> None:
        split_test = self.step_params.get('split_test', False)
        if split_test:
            self.test_objects: Optional[List[str]] = self.step_params.get("test_objects", None)
            if not self.test_objects:
                # Set required train test split method
                num_test_objects = self.step_params.get("num_test_objects", 1)
                self.test_objects = [
                    Path(file_path).stem \
                    for file_path in random.sample(self._input_files, num_test_objects)
                ]
                self.step_params["test_objects"] = self.test_objects
            else:
                self.step_params["test_objects"] = self.test_objects
        else:
            self.test_objects = []

    def _log_groups_mapping(self) -> None:
        self.file_name_mapping: Dict[str, int] = {
            Path(file_path).stem.replace(" ", "").upper(): number \
                for number, file_path in enumerate(self._input_files)
        }
        self.task.upload_artifact("group_mapping", self.file_name_mapping)

    def _concatenate_dataframes(self) -> None:
        train = pd.DataFrame()
        test = pd.DataFrame()
        try:
            for file_path in tqdm(self._input_files, total=len(self._input_files)):
                file_name = Path(file_path).stem
                self.task.logger.report_text(
                    f"Processing of {file_name}",
                    level=logging.DEBUG,
                    print_console=False,
                )
                data = PickleLoader(path=file_path).load()
                data[GROUP_ID.name] = self.file_name_mapping[file_name]

                if file_name in self.test_objects:
                    test = pd.concat([test, data])
                else:
                    train = pd.concat([train, data])

                del data
                gc.collect()

        except Exception as exception:
            self._log_failed_step_execution(
                file_name=file_name,
                exception=exception,
            )
            raise PipelineExecutionError

        self._save_locally_data(
            path=Path(os.path.join(self._output_directory, f"train{GENERAL_EXTENSION}")),
            data=train,
        )

        if self.step_params.get('split_test', False) and not test.empty:
            self._save_locally_data(
                path=Path(os.path.join(self._output_directory, f"test{GENERAL_EXTENSION}")),
                data=test,
            )

    def start(self, dataset_id: str) -> str:
        if is_empty_dir(self._input_directory):
            self._download_input_dataset(dataset_id=dataset_id)

        self._set_test_objects()
        self._log_groups_mapping()
        self._concatenate_dataframes()

        return self._upload_output_dataset(parent_datasets=[dataset_id])

