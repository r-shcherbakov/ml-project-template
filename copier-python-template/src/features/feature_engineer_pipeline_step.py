# -*- coding: utf-8 -*-
import os
import gc
import logging
from pathlib import Path
import traceback
from typing import Optional, Tuple
import warnings

import pandas as pd

from src.common.constants import GENERAL_EXTENSION
from src.common.exceptions import PipelineExecutionError
from src.common.pipeline_steps import FEATURE_ENGINEER
from src.core import BasePipelineStep
from src.features.feature_engineer import FeatureEngineer
from src.utilities.loaders import PickleLoader
from src.utilities.path_utils import is_empty_dir

warnings.simplefilter(action="ignore", category=FutureWarning)


class FeatureEngineerPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(FEATURE_ENGINEER)

    def _get_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        output = {}
        for data_type in ["train", "test"]:
            data_directory = Path(os.path.join(
                self._input_directory,
                f"{data_type}{GENERAL_EXTENSION}"
            ))
            try:
                data = PickleLoader(path=data_directory).load()
            except FileNotFoundError:
                data = None

            output[data_type] = data

        return output.get("train"), output.get("test", pd.DataFrame())

    def start(self, dataset_id: str) -> Tuple['FeatureEngineer', str]:
        if is_empty_dir(self._input_directory):
            self._download_input_dataset(dataset_id=dataset_id)

        train, test = self._get_data()
        try:
            fe = FeatureEngineer()
            fe.fit(train)
        except Exception as exception:
            self.task.logger.report_text(
                f"Featute Engineer fitting failed due to: {exception}",
                level=logging.INFO
            )
            self.task.logger.report_text(
                'traceback:' + traceback.format_exc(),
                level=logging.DEBUG,
                print_console=False,
            )
            raise PipelineExecutionError

        train_features = fe.transform(train)
        train_output_directory = Path(os.path.join(
            self._output_directory,
            f"train{GENERAL_EXTENSION}"
        ))
        self._save_locally_data(
            path=train_output_directory,
            data=train_features,
        )

        if not test.empty:
            test_features = fe.transform(test)
            test_output_directory = Path(os.path.join(
                self._output_directory,
                f"test{GENERAL_EXTENSION}"
            ))
            self._save_locally_data(
                path=test_output_directory,
                data=test_features,
            )

        del train, test, train_features, test_features
        gc.collect()

        output_dataset_id = self._upload_output_dataset(parent_datasets=[dataset_id])
        return fe, output_dataset_id
