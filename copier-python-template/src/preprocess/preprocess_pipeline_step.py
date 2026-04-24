# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
import logging
from pathlib import Path
from typing import List, Union
import warnings

from pydantic import BaseModel
from sklearn import set_config
from sklearn.pipeline import Pipeline

from src.common.constants import GENERAL_EXTENSION
from src.common.pipeline_steps import PREPROCESS
from src.core import BasePipelineStep
from src.utilities.loaders import CsvLoader
from src.utilities.path_utils import is_empty_dir
from src.preprocess.preprocessor import Preprocessor, MarkDataTransformer

warnings.simplefilter(action="ignore", category=FutureWarning)


class PreprocessParams(BaseModel):
    skip_mark: bool = False


class PreprocessPipelineStep(BasePipelineStep):
    def __init__(self, params: PreprocessParams):
        super().__init__(PREPROCESS, params=params)

    @property
    def _input_files(self) -> List[Path]:
        input_directory = self._input_directory
        file_type = r"/*csv"
        input_filepath_files = [
            Path(file_path) for file_path in glob(str(input_directory) + file_type)
        ]
        return input_filepath_files

    def _transform_input_data(self, file_path: Union[Path, str]) -> None:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        file_name = Path(file_path).stem.replace(" ", "").upper()
        self.task.logger.report_text(
            f"Processing of {file_name}",
            level=logging.DEBUG,
            print_console=False,
        )

        data = CsvLoader(path=file_path).load()
        if self.step_params.skip_mark:
            step_pipeline = Pipeline(
                [
                    ("preprocessor", Preprocessor())
                 ]
            )
        else:
            step_pipeline = Pipeline(
                steps=[
                    ("preprocessor", Preprocessor()),
                    ("add_target", MarkDataTransformer()),
                 ]
            )
        set_config(transform_output="pandas")

        try:
            preprocessed = step_pipeline.transform(data)
            self._log_success_step_execution(file_name=file_name)
        except Exception as exception:
            self._log_failed_step_execution(
                file_name=file_name,
                exception=exception,
            )

        preprocessed_filepath = Path(
            os.path.join(
                self._output_directory, f"{file_name}{GENERAL_EXTENSION}"
            )
        )
        self._save_locally_data(
            path=preprocessed_filepath,
            data=preprocessed,
        )

        del preprocessed, data
        gc.collect()

    def start(self, dataset_id: str) -> str:
        if is_empty_dir(self._input_directory):
            self._download_input_dataset(dataset_id=dataset_id)

        for path in self._input_files:
            self._transform_input_data(path)

        return self._upload_output_dataset(parent_datasets=[dataset_id])
