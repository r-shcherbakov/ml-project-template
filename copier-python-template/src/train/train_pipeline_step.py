# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
import logging
from pathlib import Path
from typing import List, Union
import warnings

from catboost import (
    Pool,
    CatBoostClassifier,
    eval_metric
)
from clearml import OutputModel
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from core.pipeline_step import BasePipelineStep
from common.exceptions import PipelineExecutionError
from common.pipeline_steps import TRAIN
from common.constants import GENERAL_EXTENSION, IGNORED_FEATURES
from preprocess.loaders import PickleLoader
from settings import Settings
from utilities.utils import (
    is_empty_dir,
    get_last_modified,
    compress_pickle,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


class TrainPipelineStep(BasePipelineStep):
    def __init__(
        self,
        settings: 'Settings'
    ):
        self.pipeline_step = TRAIN
        super().__init__(settings, self.pipeline_step)
        
    @property 
    def _get_input_directory(self) -> Path:
        return self.settings.storage.train_folder
        
    @property 
    def _get_input_files(self) -> List[Path]:
        return []
    
    @property 
    def _get_output_directory(self) -> Path:
        return self.settings.storage.prediction_folder
    
    def _upload_artifacts(self) -> None:
        pass
    
    def _get_data(self, path: Union[Path, str]) -> pd.DataFrame:
        file_path = get_last_modified(path=path, suffixes=GENERAL_EXTENSION)
        data = PickleLoader(path=file_path).load()
        return data
    
    def _set_ignored_features(self):
        ignored_features = list(np.intersect1d(self.train_data.columns.tolist(), IGNORED_FEATURES))
        self.step_params["ignored_features"] = ignored_features
        
    def _get_cb_pool(self, data: pd.DataFrame) -> Pool:
        X = data.drop(axis=1, columns="TARGET").copy()
        y = data["TARGET"].fillna(0).copy()
        groups = data["GROUP_ID"].copy()
        cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        pool = Pool(
            data=X,
            label=y,
            group_id=groups,
            cat_features=cat_features,
        )
        return pool
        
    def _run_cv(self):
        """
        Uploaded results of common metrics for each fold 
        provided by GroupKFold cross-validation.
        """
        
        X = self.train_data.drop(axis=1, columns="TARGET").copy()
        y = self.train_data["TARGET"].fillna(0).copy()
        groups = self.train_data["GROUP_ID"].copy()
        
        n_splits = self.step_params.pop("n_splits", 2)
        skf = GroupKFold(n_splits=n_splits)
        cv_result = pd.DataFrame()
        for train_idx, valid_idx in tqdm(skf.split(X, y, groups=groups), total=n_splits):
            self.task.logger.report_text(
                f"Train GROUP_ID:{X.loc[train_idx, 'GROUP_ID'].unique()}",
                level=logging.DEBUG,
                print_console=False,
            )
            self.task.logger.report_text(
                f"Test GROUP_ID:{X.loc[valid_idx, 'GROUP_ID'].unique()}",
                level=logging.DEBUG,
                print_console=False,
            )
            
            train_pool = self._get_cb_pool(self.train_data[train_idx])
            eval_pool = self._get_cb_pool(self.train_data[valid_idx])
            fitted_model = CatBoostClassifier(
                **self.step_params,
                random_seed=self.settings.random_seed,
            ).fit(
                train_pool,
                eval_set=eval_pool,
            )

            fold_result = {}
            metrics = ["Accuracy", "AUC", "Recall", "Precision", "F1", "BalancedAccuracy", "MCC"]
            for metric in metrics:
                fold_result[f"{metric}"] = fitted_model.eval_metrics(eval_pool, [metric])[
                    metric
                ][-1]
            fold_result = pd.DataFrame([fold_result])
            cv_result = pd.concat([cv_result, fold_result])
            del train_pool, eval_pool
            gc.collect()
            
        cv_result.loc["mean"] = cv_result.mean()
        self.task.logger.report_table(
            title="CV results", 
            series="CV results",
            table_plot=cv_result
        ) 
        
    def _train_model(self):
        train_pool = self._get_cb_pool(self.train_data)
        test_pool = self._get_cb_pool(self.test_data)

        self.fitted_model = CatBoostClassifier(
            **self.step_params,
            random_seed=self.settings.random_seed,
        ).fit(train_pool, eval_set=test_pool, verbose=True)
        
        fitted_model_filepath = os.path.join(self.settings.artifacts.models_folder, "example.cbm")
        self.fitted_model.save_model(fitted_model_filepath)
        
        self.output_model = OutputModel(task=self.task)
        self.output_model.update_labels(train_pool.get_label())
        self.output_model.update_weights(weights_filename=fitted_model_filepath)
        
    def _get_prediction(self) -> pd.DataFrame:
        # Get prediction with fitted models
        prediction = pd.DataFrame()
        prediction["GROUP_ID"] = self.test_data["GROUP_ID"].copy()
        probability = np.empty(len(self.test_data))
        try: 
            # In case of several fitted models iterate over it
            models = [self.fitted_model]
            for model in models:
                required_columns = model.feature_names_
                probability += pd.DataFrame(
                    model.predict(self.test_data[required_columns], prediction_type="Probability")
                )[1]
            probability = probability / len(models)
            prediction["PREDICTION_CONT"] = probability
            
            threshold = self.step_params.get("binary_threshold", 0.5)
            prediction["PREDICTION_DISC"] = np.where(probability > threshold, 1, 0)
        except Exception as exception:
            self._log_failed_step_execution(
                file_name="test_pool",
                exception=exception
            )
            raise PipelineExecutionError
        
        return prediction
        
    def _predict_test(self):
        prediction = self._get_prediction()
        
        # Save locally predictions for plots
        self._save_locally_data(
            path=self._get_output_directory,
            data=prediction,
        )
            
        # Track metrics at test dataset
        test_metrics = {}
        metrics = ["Accuracy", "AUC", "Recall", "Precision", "F1", "BalancedAccuracy", "MCC"]
        for metric in metrics:
            test_metrics[f"{metric}"] = eval_metric(
                label=self.test_data["TARGET"].fillna(0).copy(),
                approx=prediction["PREDICTION_DISC"],
                metric=metric,
            )[0]
        test_metrics = pd.DataFrame.from_dict(test_metrics)
        self.task.logger.report_table(
            title="test metrics", 
            series="test metrics",
            table_plot=test_metrics
        ) 
        
    def _process_data(self) -> None:
        self.train_data = self._get_data(self._get_input_directory)
        self.test_data = self._get_data(self.settings.storage.test_folder)
        self._set_ignored_features()
        
        # Cross validation
        self._run_cv()
        
        # Train final model
        self._train_model()
        
        # Get prediction for test data
        self._predict_test()
        
        

        
        