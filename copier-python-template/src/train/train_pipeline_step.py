# -*- coding: utf-8 -*-
import os
import gc
import logging
from pathlib import Path
from typing import Optional, Tuple
import warnings

from catboost import (
    Pool,
    CatBoostClassifier,
    eval_metric
)
import numpy as np
import pandas as pd
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


class TrainPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(TRAIN)

    @property
    def _metrics(self) -> list[str]:
        return ["Precision", "Recall", "F1", "BalancedAccuracy",  "AUC"]

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
        ignored_features = list(np.intersect1d(self.train_data.columns.tolist(), IGNORED_FEATURES))
        self.step_params["ignored_features"] = ignored_features

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
        """
        Uploaded results of common metrics for each fold
        provided by GroupKFold cross-validation.
        """

        X = self.train_data.drop(axis=1, columns=TARGET.name).copy()
        y = self.train_data[TARGET.name].fillna(0).copy()
        groups = self.train_data[GROUP_ID.name].copy()

        n_splits = self.step_params.pop("n_splits", 2)
        skf = GroupKFold(n_splits=n_splits)
        cv_result = pd.DataFrame()
        for train_idx, valid_idx in tqdm(skf.split(X, y, groups=groups), total=n_splits):
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
            **self.step_params,
            random_seed=self.settings.random_seed,
        ).fit(train_pool, eval_set=test_pool, verbose=True)

        fitted_model_filepath = os.path.join(self.settings.artifacts.models_folder, "example.cbm")
        self.fitted_model.save_model(
            fitted_model_filepath,
            format="cbm"
        )

    def _get_prediction(self) -> pd.DataFrame:
        # Get prediction with fitted models
        prediction = pd.DataFrame()
        prediction[GROUP_ID.name] = self.test_data[GROUP_ID.name].copy()
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
            prediction[PROBABILITY_PREDICTION.name] = probability

            threshold = self.step_params.get("binary_threshold", 0.5)
            prediction[DISCRETE_PREDICTION.name] = np.where(probability > threshold, 1, 0)
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
            path=self._output_directory,
            data=prediction,
        )

        # Track metrics at test dataset
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

        # Cross validation
        self._run_cv()

        # Train final model
        self._train_model()

        # Get prediction for test data
        if not self.test_data.empty:
            self._predict_test()




