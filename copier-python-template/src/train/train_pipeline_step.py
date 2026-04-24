# -*- coding: utf-8 -*-
import os
import gc
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

from catboost import (
    Pool,
    CatBoostClassifier,
    eval_metric
)
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.core import BasePipelineStep
from src.common.exceptions import PipelineExecutionError
from src.common.pipeline_steps import TRAIN
from src.common.constants import GENERAL_EXTENSION
from src.common.features import (
    GROUP_ID,
    IGNORED_FEATURES,
    TARGET,
    DISCRETE_PREDICTION,
    PROBABILITY_PREDICTION
)
from src.utilities.loaders import PickleLoader
from src.utilities.path_utils import is_empty_dir

warnings.simplefilter(action="ignore", category=FutureWarning)


class TrainParams(BaseModel):
    skip_cv: bool = True
    n_splits: int = 4
    train_final_model: bool = True
    binary_threshold: float = 0.5


class TrainPipelineStep(BasePipelineStep):
    def __init__(self, params: TrainParams):
        super().__init__(TRAIN, params=params)

    @property
    def _metrics(self) -> list[str]:
        return ["Precision", "Recall", "F1", "BalancedAccuracy", "AUC"]

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

    def _set_ignored_features(self):
        self.ignored_features: List[str] = list(
            np.intersect1d(self.train_data.columns.tolist(), IGNORED_FEATURES)
        )

    def _get_cb_pool(self, data: Optional[pd.DataFrame]) -> Pool:
        if not data.empty:
            X = data.drop(axis=1, columns=TARGET.name).copy()
            y = data[TARGET.name].fillna(0).copy()
            groups = data[GROUP_ID.name].copy()
            cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

            pool = Pool(
                data=X,
                label=y,
                group_id=groups,
                cat_features=cat_features,
            )
            return pool
        else:
            return None

    def _run_cv(self):
        X = self.train_data.drop(axis=1, columns=TARGET.name).copy()
        y = self.train_data[TARGET.name].fillna(0).copy()
        groups = self.train_data[GROUP_ID.name].copy()

        skf = GroupKFold(n_splits=self.step_params.n_splits)
        cv_result = pd.DataFrame()
        for train_idx, valid_idx in tqdm(skf.split(X, y, groups=groups), total=self.step_params.n_splits):
            self.task.logger.report_text(
                f"Train GROUP_ID:{X.loc[train_idx, GROUP_ID.name].unique()}",
                level=logging.DEBUG,
                print_console=False,
            )
            self.task.logger.report_text(
                f"Test GROUP_ID:{X.loc[valid_idx, GROUP_ID.name].unique()}",
                level=logging.DEBUG,
                print_console=False,
            )

            train_pool = self._get_cb_pool(self.train_data.iloc[train_idx])
            eval_pool = self._get_cb_pool(self.train_data.iloc[valid_idx])
            fitted_model = CatBoostClassifier(
                random_seed=self.settings.random_seed,
            ).fit(
                train_pool,
                eval_set=eval_pool,
            )

            fold_result = {}
            for metric in self._metrics:
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
            random_seed=self.settings.random_seed,
        ).fit(train_pool, eval_set=test_pool, verbose=True)

        fitted_model_filepath = os.path.join(self.settings.artifacts.models_folder, "example.cbm")
        self.fitted_model.save_model(
            fitted_model_filepath,
            format="cbm"
        )

    def _get_prediction(self) -> pd.DataFrame:
        prediction = pd.DataFrame()
        prediction[GROUP_ID.name] = self.test_data[GROUP_ID.name].copy()
        probability = np.empty(len(self.test_data))
        try:
            models = [self.fitted_model]
            for model in models:
                required_columns = model.feature_names_
                probability += pd.DataFrame(
                    model.predict(self.test_data[required_columns], prediction_type="Probability")
                )[1]
            probability = probability / len(models)
            prediction[PROBABILITY_PREDICTION.name] = probability

            prediction[DISCRETE_PREDICTION.name] = np.where(
                probability > self.step_params.binary_threshold, 1, 0
            )
        except Exception as exception:
            self._log_failed_step_execution(
                file_name="test_pool",
                exception=exception
            )
            raise PipelineExecutionError

        return prediction

    def _predict_test(self):
        prediction = self._get_prediction()

        self._save_locally_data(
            path=self._output_directory,
            data=prediction,
        )

        test_metrics = {}
        for metric in self._metrics:
            test_metrics[f"{metric}"] = eval_metric(
                label=self.test_data[TARGET.name].fillna(0).copy(),
                approx=prediction[DISCRETE_PREDICTION.name],
                metric=metric,
            )[0]
        test_metrics = pd.DataFrame.from_dict(test_metrics)
        self.task.logger.report_table(
            title="test metrics",
            series="test metrics",
            table_plot=test_metrics
        )

    def start(self, dataset_id: str) -> None:
        if is_empty_dir(self._input_directory):
            self._download_input_dataset(dataset_id=dataset_id)

        self.train_data, self.test_data = self._get_data()
        self._set_ignored_features()

        if not self.step_params.skip_cv:
            self._run_cv()

        if self.step_params.train_final_model:
            self._train_model()

        if not self.test_data.empty:
            self._predict_test()
