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

from common.constants import GENERAL_EXTENSION
from common.exceptions import PipelineExecutionError
from common.pipeline_steps import SPLIT_DATASET
from core.pipeline_step import BasePipelineStep
from preprocess.loaders import PickleLoader
from settings import Settings
from utilities.utils import is_empty_dir

warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitDatasetPipelineStep(BasePipelineStep):
    def __init__(
        self,
        settings: 'Settings'
    ):
        self.pipeline_step = SPLIT_DATASET
        super().__init__(settings, self.pipeline_step)
        
    @property 
    def _get_input_directory(self) -> Path:
        return self.settings.storage.features_folder
        
    @property 
    def _get_input_files(self) -> List[Path]:
        input_directory = self._get_input_directory
        if is_empty_dir(input_directory):
            self._download_input_dataset()
        else:
            self._upload_input_dataset()
              
        file_type = f"/*{GENERAL_EXTENSION}"
        input_filepath_files = [
            Path(file_path) for file_path in glob(str(input_directory) + file_type)
        ]
        return input_filepath_files
    
    @property 
    def _get_output_directory(self) -> Path:
        return self.settings.storage.splitted_folder
    
    def _upload_artifacts(self) -> None:
        pass
    
    def _set_test_objects(self) -> None:
        split_test = self.step_params.get('split_test', False)
        if split_test:
            self.test_objects: Optional[List[str]] = self.step_params.get("test_objects", None)
            if self.test_objects is None:
                random.seed(self.settings.random_seed)
                num_test_objects = self.step_params.get("num_test_objects", 1)
                self.test_objects = [
                    Path(file_path).stem \
                    for file_path in random.sample(self._get_input_files, num_test_objects)
                ]
                self.step_params["test_objects"] = self.test_objects
            else:
                self.step_params["test_objects"] = self.test_objects
        else:
            self.test_objects = []
            
    def _log_groups_mapping(self) -> None:
        self.file_name_mapping: Dict[str, int] = {
            Path(file_path).stem.replace(" ", "").upper(): number \
                for number, file_path in enumerate(self._get_input_files)
        }
        self.task.upload_artifact("groups_mapping", self.file_name_mapping)  
    
    def _process_data(self) -> None:
        self._set_test_objects()
        self._log_groups_mapping()
            
        train = pd.DataFrame()
        test = pd.DataFrame()
        try:
            for file_path in tqdm(self._get_input_files, total=len(self._get_input_files)):
                file_name = Path(file_path).stem
                self.task.logger.report_text(
                    f"Processing of {file_name}", 
                    level=logging.DEBUG,
                    print_console=False,
                )
                data = PickleLoader(path=file_path).load()
                data['GROUP_ID'] = self.file_name_mapping[file_name]

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
            path=self.settings.storage.train_folder,
            data=train,
        )

        if self.step_params.get('split_test', False) and not test.empty:
            self._save_locally_data(
                path=self.settings.storage.test_folder,
                data=test,
            )