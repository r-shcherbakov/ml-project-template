import importlib
import os
from typing import List

from src.config import target_curves, DATA_PATH, INDICES_DIR
from src.core.preprocess import Preprocess
from src.core.stratification import Stratification
from src.utils.mapping import MnemonicDictionary


class Test:
    """
    Class for testing model, loading data, computing metrics

    Args:
        cfg: dict with config
        metrics: list of metrics to compute
        targets: list of targets curves
        mode: feature set mode

    Attributes:
        cfg: dict with config
        mode: feature set mode
        strat: Stratification object for loading data
        targets: list of target curves
        scaler: scaler object for data normalization
        feature_curves: list of feature curves
        dataset: dataset for testing
        metrics: list of metrics to compute
        metrics_objs: dict with metrics objects
    """
    def __init__(self, cfg: dict, metrics: List[str], targets: List[str], mode: int):
        self.cfg = cfg
        self.mode = mode
        self.strat = Stratification(DATA_PATH, mode=mode)
        self.targets = targets
        self.scaler = None
        self.feature_curves = MnemonicDictionary(use_rop=cfg['prepare']['use_rop']).sets[mode]
        self.dataset = None
        self.metrics = metrics
        self.metrics_objs: dict = {}

        if self.cfg['prepare']['normalize']:
            self.scaler = Preprocess.load_scaler(os.path.join(INDICES_DIR, str(mode), 'scaler.npy'))

        self.load_data()
        self.init_metrics()

        if self.cfg['prepare']['create_features']:
            feature_curves = list(self.dataset[-1][1][0].columns.values)
            for target in target_curves:
                feature_curves.remove(target)

    def load_data(self):
        """
        Load dataset for testing
        """
        self.dataset = self.strat.get_dataset_by_indices(is_train=False)

    def init_metrics(self):
        """
        Initialize metrics objects

        Raises:
            ModuleNotFoundError: if metric is not found
        """
        for metric in self.metrics:
            try:
                module_ = importlib.import_module(f'src.core.metrics', metric)
                class_ = getattr(module_, metric)
                self.metrics_objs[metric] = class_()
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f'Metric {metric} not found')

    def compute_metrics(self, y_true, y_pred) -> dict:
        """
        Compute metrics for given test and predicted values.
        Args:
            y_true: test values
            y_pred: predicted values
        Returns:
            dict with results
        """
        metrics = {}
        for metric, metric_obj in self.metrics_objs.items():
            metrics[metric] = metric_obj(y_true, y_pred)
        return metrics

    def get_results(self) -> dict:
        """
        Get all calculated results for all metrics in dict.
        Results are calculated by mean function specified in each Metric object.
        Returns:
            dict with results
        """
        results = {}
        for metric in self.metrics:
            results[metric] = self.metrics_objs[metric].get_results()
        return results

    def repr_results(self) -> str:
        """
        Get formatted string representation of results.
        Returns:
            string representation of results

        """
        return ', '.join([f'\t{metric}: {metric_res:.4f}'for metric, metric_res in self.get_results().items()])
