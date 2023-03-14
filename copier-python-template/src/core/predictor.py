from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import torch
from clearml import Task
from torch.utils.data import DataLoader

from src.config import target_curves
from src.utils.dataloader import SequenceDataset
from src.utils.mapping import MnemonicDictionary
from src.utils.utils import prepare_parameter_name

mnem_dict = MnemonicDictionary()


class Predictor:
    """
    Class that make and save predictions for torch models

    Attributes:
        model (torch.nn.Module): model to make predictions
        device (str): device to use
        config (dict): config to use
        dir_to_save (str): directory to save predictions
    """
    def __init__(self, model: torch.nn.Module = None, model_path: str = None, config: dict = None,
                 dir_to_save: str = None):
        self.model = model
        self.device = config['general']['device']
        self.model.to(self.device)
        self.config = config
        self.dir_to_save = dir_to_save
        if model is None:
            self._load_weights(model_path)

    def _load_weights(self, path: str):
        """
        Load model weights from file
        Args:
            path (str): path to file with model weights
        """
        self.model = joblib.load(path)

    def predict(self, x: np.ndarray) -> pd.DataFrame:
        """
        Make predictions for input data
        Args:
            x (np.ndarray): input data

        Returns:
            pd.DataFrame: predictions
        """
        test_dataset = SequenceDataset().init_x_y(
            x,
            sequence_length=self.config['train']['sequence_length'],
            output_length=self.config['train']['output_length'],
        )

        test_loader = DataLoader(test_dataset, batch_size=self.config['train']['batch_size'],
                                 shuffle=False)

        self.model.eval()
        acc_results = [None] * (len(x) + self.config['train']['output_length'])
        with torch.no_grad():
            for i, (X, y, ext) in enumerate(test_loader):
                predict = self.model(X).detach().to('cpu').numpy()
                for j, i_step in zip(range(len(predict)), ext):
                    for k in range(self.config['train']['output_length']):
                        if acc_results[i_step + k] is None:
                            acc_results[i_step + k] = [predict[j, k]]
                        else:
                            acc_results[i_step + k].append(predict[j, k])
            weigths = [i / self.config['train']['output_length']
                       for i in range(1, self.config['train']['output_length'] + 1)]
        for i in range(len(acc_results)):
            if acc_results[i] is None:
                acc_results[i] = [0] * self.config['train']['output_length']
            else:
                acc_results[i] = np.average(acc_results[i], axis=0, weights=weigths[-len(acc_results[i]):])
        acc_results = pd.DataFrame(acc_results[:len(x)])

        if self.config['inference']['clip_predictions']:
            mnems_bounds = {mn: mnem_dict.get_limits(mn) for mn in target_curves}

            acc_results = clip_predictions(acc_results, mnems_bounds)

        return acc_results

    def save_prediction(self, filename: str, pred: pd.DataFrame):
        """
        Save prediction to file at self.dir_to_save
        Args:
            filename (str): name of file to save
            pred (pd.DataFrame): prediction to save
        """
        os.makedirs(self.dir_to_save, exist_ok=True)

        np.save(os.path.join(self.dir_to_save, prepare_parameter_name(filename)), pred)
        task = Task.current_task()
        if task:
            task.upload_artifact(prepare_parameter_name(f'Predictions_{filename}'), pred)

    def predict_and_save(self, x: np.ndarray, filename: str):
        """
        Make prediction and save it to file
        Args:
            x (np.ndarray): input data
            filename (str): name of file to save
        """
        pred = self.predict(x)
        self.save_prediction(filename, pred)


def clip_predictions(pred: pd.DataFrame, mnems_bounds: dict[str, list[int, int]]) -> pd.DataFrame:
    """
    Clip predictions to bounds
    Args:
        pred (pd.DataFrame): predictions DataFrame
        mnems_bounds (dict[list[int, int]]): bounds for each mnemonic that exist in predictions

    Returns:
        pd.DataFrame: clipped predictions
    """
    for i, (mn, mn_bounds) in enumerate(mnems_bounds.items()):
        pred.iloc[:, i] = pred.iloc[:, i].clip(mn_bounds[0], mn_bounds[1])
    return pred
