import itertools
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from clearml import Task

from src.config import INDICES_DIR, test_filename, train_filename, val_filename
from src.core.parser import Parser
from src.core.preprocess import Preprocess
from src.utils.clusterization import clustering
from src.utils.mapping import MnemonicDictionary


class Stratification:
    def __init__(self, data_path: str, test_part: float = 0.1, normalize: bool = False,
                 extra_features: List = None, feature_map: MnemonicDictionary = None, mode: int = 0,
                 cross_validation: bool = False):
        self._parser = Parser()
        self._data_path = data_path
        self._normalize = normalize
        self._extra_features = extra_features
        self._test_part = test_part
        self._data: Optional[dict] = None
        self.mode = mode
        self.cross_validation = cross_validation
        self._feature_map = feature_map
        self._task: Task = Task.current_task()

        self._load_data()

    def _load_data(self):
        self._data = self._parser.load_obj(self._data_path)
        return self._data

    def get_data_df(self, data_df: pd.DataFrame):
        if self._normalize:
            data_df = Preprocess.normalize_df(data_df, self.mode)
        if self._extra_features:
            extra_df = pd.concat([data_df.pop(x) for x in self._extra_features], axis=1)
            data_df = pd.concat([data_df, extra_df], axis=1)
        return data_df

    def strat_by_location(self, val: float = None):
        clusters, outliers = clustering(self._data)
        train_idx, test_idx, val_idx = self.get_dataset_indices(clusters, outliers, val=val)

        print(f"Total length: {len(self._data)}, "
              f"train len: {len(train_idx)}, "
              f"test len: {len(test_idx)}, "
              f"val len: {len(val_idx) if not val_idx is None else val_idx}")

        np.save(os.path.join(INDICES_DIR, str(self.mode), train_filename), train_idx)
        np.save(os.path.join(INDICES_DIR, str(self.mode), test_filename), test_idx)
        if val_idx:
            np.save(os.path.join(INDICES_DIR, str(self.mode), val_filename), val_idx)

    def strat_by_location_cv(self, val: float = None):
        """
        Stratification for cross validation
        """
        clusters, outliers = clustering(self._data)
        folds_num = round(1 / val)
        folds = {f'fold_{i}': f'fold_{i}.npy' for i in range(folds_num)}
        folds_indices = self.get_dataset_indices_cv(clusters, outliers, folds)
        print(f"Total length: {len(self._data)}")
        # save folds indices
        for fold, indices in folds_indices.items():
            np.save(os.path.join(INDICES_DIR, str(self.mode), folds[fold]), indices)
            print(f"Fold {fold} length: {len(indices)}")
        

    def get_dataset_indices_cv(self, clusters, outliers, folds: dict):
        # distribute clusters items evenly between folds
        folds_num = len(folds)
        # infinite loop over folds dictionary
        fold_gen = itertools.cycle(folds.keys())
        folds_items = {fold: [] for fold in folds.keys()}
        # make outliers struecture the same as clusters
        outliers = [np.array([el]) for el in outliers]
        for cluster in clusters + outliers:
            for item in cluster:
                fold = next(fold_gen)
                folds_items[fold].append(np.array(item))
        return folds_items



    def get_dataset_indices(self, clusters, outliers, val: float = None):
        train_idx, test_idx, val_idx = [], [], []

        for cluster in clusters:
            test_idx_count = max(1, round(len(cluster) * self._test_part)) if self._test_part else 0
            if test_idx_count != 0:
                test_idx.append(cluster[-test_idx_count:])
                train_idx.append(cluster[:-test_idx_count])
            else:
                train_idx.append(cluster)
            if val is not None:
                val_idx_count = max(1, round(len(train_idx[-1]) * val)) if val else 0
                if val_idx_count == 0:
                    val_idx.append([])
                    continue
                val_idx.append(train_idx[-1][-val_idx_count:])
                train_idx[-1] = train_idx[-1][:-val_idx_count]

        if outliers:
            test_idx_count = round(len(outliers) * self._test_part) if self._test_part else 0
            if test_idx_count != 0:
                test_idx.append(outliers[-test_idx_count:])
                train_idx.append(outliers[:-test_idx_count])
            else:
                train_idx.append(outliers)
            if val is not None:
                val_idx_count = round(len(train_idx[-1]) * val) if val else 0
                if val_idx_count == 0:
                    val_idx.append([])
                else:
                    val_idx.append(train_idx[-1][-val_idx_count:])
                    train_idx[-1] = train_idx[-1][:-val_idx_count]

        return list(itertools.chain.from_iterable(train_idx)), list(itertools.chain.from_iterable(test_idx)), \
               list(itertools.chain.from_iterable(val_idx)) if val_idx else None

    def get_dataset_by_indices(self, is_train: bool = True, need_val: bool = False, path_to_indices=INDICES_DIR):
        self._folds = Parser.load_obj(os.path.join(path_to_indices, str(self.mode), 'data'))

        if is_train:
            train_idx = np.load(os.path.join(path_to_indices, str(self.mode), train_filename), allow_pickle=True)
            train = pd .concat([self.get_fold_by_well(idx, use_mask=False) for idx in train_idx])
            if need_val and os.path.exists(os.path.join(path_to_indices, str(self.mode), val_filename)):
                val_idx = np.load(os.path.join(path_to_indices, str(self.mode), val_filename), allow_pickle=True)
                val = pd.concat([self.get_fold_by_well(idx, use_mask=False) for idx in val_idx])
            else:
                val = None

            if self._normalize:
                if val is not None:
                    data_df = pd.concat([part for part in [train, val]])
                    data_df = self.get_data_df(data_df)
                    return data_df.loc[train.index], data_df.loc[val.index]
                else:
                    return self.get_data_df(train), val
            return train, val
        else:
            test_idx = np.load(os.path.join(path_to_indices, str(self.mode), test_filename), allow_pickle=True)
            if test_idx.size == 0:
                test_idx = np.load(os.path.join(path_to_indices, str(self.mode), val_filename), allow_pickle=True)
            test = [('_'.join(idx), self.get_fold_by_well(idx, is_test=True)) for idx in test_idx]
            return test

    def get_fold_by_well(self, item, use_mask: bool = False, is_test: bool = False, use_plast: bool = True):
        res = self._folds[item[1]]['data']
        mask = self._folds[item[1]]['mask']
        if use_plast and use_mask:
            if not (isinstance(mask, pd.DataFrame) and 'BTH' in mask.columns):
                raise ValueError(f'Variable mask is not the instance of pd.DataFrame or BTH column doesnt exist.')
            mask = mask.mask(~mask['BTH'], False)

        if is_test:
            return res, mask
        elif use_mask:
            return self.use_filter_mask(res, mask)

        return res

    @staticmethod
    def process_nan(df: pd.DataFrame, strategy: str = 'drop'):
        if strategy == 'drop':
            df = df.dropna()
        else:
            print(f'Unknown processing nans strategy {strategy}!')
        return df

    def use_filter_mask(self, data_df: pd.DataFrame, mask: pd.DataFrame):
        # set NaN if data is out of limits
        data_df = data_df.mask(~mask)
        data_df = self.process_nan(data_df)

        return data_df


def update_train_val_from_folds(config: dict, val_fold_num: int, silent: bool = False):
    mode = int(config['prepare']['feature_model'])
    folds_files = {int(re.findall("\d+", file)[0]): file for file in os.listdir(
                   os.path.join(INDICES_DIR, str(mode))) 
                   if file.startswith('fold_')}
    folds_items = {k: np.load(os.path.join(INDICES_DIR, str(mode), v), allow_pickle=True) 
                   for k, v in folds_files.items()}
    train_idx = np.concatenate([v for k, v in folds_items.items() if k != val_fold_num])
    val_idx = folds_items[val_fold_num]
    np.save(os.path.join(INDICES_DIR, str(mode), train_filename), train_idx)
    np.save(os.path.join(INDICES_DIR, str(mode), val_filename), val_idx)
    np.save(os.path.join(INDICES_DIR, str(mode), test_filename), np.array([]))
    if not silent:
        print(f'Updated train and val indices for fold {val_fold_num}.')
    