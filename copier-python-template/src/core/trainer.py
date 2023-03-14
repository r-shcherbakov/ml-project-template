import json
import os

import numpy as np
import pandas as pd
import torch
from clearml import Task
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config import compile_model, extra_curves, target_curves
from src.core.metrics import MAE, R2, RMSE
from src.models import SCINet, WellInception, WellLSTM
from src.utils.adan import Adan
from src.utils.dataloader import SequenceDataset
from src.utils.train_model import MSE_OHEM
from src.utils.utils import prepare_parameter_name, torch2_compile


@torch2_compile(compile_model)
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = ss_res / ss_tot
    return r2


class Trainer:
    def __init__(self, path_to_result: str, config: dict, extra_df_train=None, extra_df_val=None,
                 extra_at_train: bool = False, feature_curves: list = None, weights: str = None,
                 block_hidden_size: int = 1):
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.batch_size = config['batch_size']
        self.sequence_length = config['sequence_length']
        self.output_length = config['output_length']
        self.bidirectional = config.get('lstm_bidirectional', False)
        self.layer_norm = config.get('lstm_layer_norm', False)
        self.lr_patience = config.get('lr_patience', 5)
        self.block_hidden_size = block_hidden_size
        self._extra_df_train = extra_df_train
        self._extra_df_val = extra_df_val
        self._extra_at_train = extra_at_train
        self.path = path_to_result
        self._losses = {"losses": []}
        self._early_stop_patience = config.get('early_stop_patience', 10)
        self._model_type = config.get('model', 'conv')
        self._ratio = config.get('loss_ratio', 0.5)
        self._task: Task = Task.current_task()
        self._metrics_objs = [RMSE(), MAE(), R2()]
        self._ix_epoch = 0

        self.load_model(feature_curves=feature_curves, weights=weights)
        self.load_train_functions(config.get('init_lr', 1e-4))

        os.makedirs(self.path, exist_ok=True)

    def load_model(self, feature_curves: list = None, weights: str = None):
        input_size = len(feature_curves)

        if self._model_type == 'lstm':
            self.model = WellLSTM(input_size, output_dimensions=len(target_curves),
                                  bidirectional=self.bidirectional,
                                  layer_norm=self.layer_norm)
            if weights is not None:
                self.model.load_state_dict(torch.load(weights))
        elif self._model_type == 'conv':
            self.model = WellInception(input_size, output_dimensions=len(target_curves))
        elif self._model_type == 'scinet':
            self.model = SCINet(output_dim=len(target_curves), input_dim=input_size, hid_size=self.block_hidden_size,
                                output_len=self.output_length, input_len=self.sequence_length,
                                modified=False, num_stacks=1, groups=1)
        else:
            raise Exception(f'Unknown model type {self._model_type}!')
        if torch.__version__.startswith('2.0') and compile_model:
            self.model = torch.compile(self.model)
        print(f'Model {type(self.model)} initialized!')
        if torch.cuda.is_available():
            self.model.to('cuda:0')

    def load_train_functions(self, start_lr: float):
        self.loss_function = MSE_OHEM(ratio=self._ratio, root=False)
        if torch.__version__.startswith('2.0') and compile_model:
            self.loss_function = torch.compile(self.loss_function)
        self.optimizer = Adan(self.model.parameters(), lr=start_lr, weight_decay=0.02,
                              max_grad_norm=1.0, no_prox=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.lr_patience)

    @staticmethod
    def load_data(dataset: pd.DataFrame, batch_size: int, is_train: bool = True, sequence_length: int = 21,
                  output_length: int = 1, feature_curves: list = None) -> DataLoader:
        train_dataset = SequenceDataset().init_df(
            dataset,
            target=target_curves,
            features=feature_curves,
            extra_features=extra_curves,
            sequence_length=sequence_length,
            output_length=output_length,
        )

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_train, num_workers=3,
                                 persistent_workers=True)

        return data_loader

    def iter_train(self, data_loader: DataLoader) -> float:
        total_loss = 0
        total_r2 = []
        self.model.train()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        for i, (X, y, ext) in enumerate(data_loader):
            starter.record()
            output = self.model(X)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)
            loss = self.loss_function(output, y)
            r2 = r2_loss(output, y)
            output_np = output.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for metric in self._metrics_objs:
                metric(output_np, y_np)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_r2.append(r2.item())

        avg_loss = total_loss / i
        print(f"Train loss: {avg_loss}; R2: {1 - np.array(total_r2).mean()}; steps={i};")
        if self._task:
            for metric in self._metrics_objs:
                self._task.get_logger().report_scalar(title=prepare_parameter_name(metric.name), 
                                                      series='Train value', value=metric.get_results(),
                                                      iteration=self._ix_epoch)
                metric.reset()
            # report inference speed
            self._task.get_logger().report_scalar(title=prepare_parameter_name('Model forward time (ms)'), 
                                                  series='Train value', value=np.sum(timings) / len(timings), 
                                                  iteration=self._ix_epoch)

        return avg_loss

    def get_test_loss(self, data_loader: DataLoader) -> float:
        total_loss = 0
        total_r2 = []
        self.model.eval()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for i, (X, y, ext) in enumerate(data_loader):
                starter.record()
                output = self.model(X)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
                cur_loss = self.loss_function(output, y).item()
                r2 = r2_loss(output, y)
                output_np = output.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                for metric in self._metrics_objs:
                    metric(output_np, y_np)
                total_loss += cur_loss
                total_r2.append(r2.item())

        avg_loss = total_loss / (i+1)
        print(f"Test loss: {avg_loss}; R2: {1 - np.array(total_r2).mean()}; steps={i}")
        if self._task:
            for metric in self._metrics_objs:
                self._task.get_logger().report_scalar(title=prepare_parameter_name(metric.name), 
                                                      series='Val value', value=metric.get_results(),
                                                      iteration=self._ix_epoch)
                metric.reset()
            self._task.get_logger().report_scalar(title=prepare_parameter_name('Model forward time (ms)'), 
                                                  series='Val value', value=np.sum(timings) / len(timings), 
                                                  iteration=self._ix_epoch)
        return avg_loss

    def run_train(self, train_loader: DataLoader, val_loader, num_epoch: int = 20):
        min_loss = np.inf
        early_stop_counter = 0
        for ix_epoch in range(num_epoch):
            self._ix_epoch = ix_epoch
            print(f"Epoch {ix_epoch}\n---------")
            train_loss = self.iter_train(train_loader)
            val_loss = self.get_test_loss(val_loader)

            if val_loss < min_loss:
                self.save_weights()
                min_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            self.scheduler.step(val_loss)
            print(f'LR={self.optimizer.param_groups[0]["lr"]}\n')

            if self._task:
                self._task.get_logger().report_scalar(title=prepare_parameter_name('Losses'), 
                                                      series='Train loss', value=train_loss,
                                                      iteration=ix_epoch)
                self._task.get_logger().report_scalar(title=prepare_parameter_name('Losses'), 
                                                      series='Val loss', value=val_loss,
                                                      iteration=ix_epoch)
                self._task.get_logger().report_scalar(title=prepare_parameter_name('Learning rate'), 
                                                      series='LR',
                                                      value=self.optimizer.param_groups[0]["lr"],
                                                      iteration=ix_epoch)

            self._losses["losses"].append({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': ix_epoch})
            if early_stop_counter >= self._early_stop_patience:
                print(f'Early stop. Val loss not improving {early_stop_counter} epochs.')
                break

    def save_weights(self):
        torch.save(self.model.state_dict(), os.path.join(self.path, 'weights.pkl'))
        if self._task is not None:
            self._task.upload_artifact(prepare_parameter_name('Model weights'), 
                                       os.path.join(self.path, 'weights.pkl'))

    def save_losses(self):
        if self._losses["losses"]:
            with open(os.path.join(self.path, 'losses.json'), 'w') as f:
                json.dump(self._losses, f)
            if self._task is not None:
                self._task.upload_artifact(prepare_parameter_name('Losses'), 
                                           os.path.join(self.path, 'losses.json'))

