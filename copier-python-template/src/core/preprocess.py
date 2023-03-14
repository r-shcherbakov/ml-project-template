import os
import pickle
import re
import shutil
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import welly
from clearml import Task
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, INDICES_DIR, min_zenit_angle
from src.core.parser import Parser
from src.utils.mapping import MnemonicDictionary


class Preprocess:
    def __init__(self, data_name: str, features: Optional[list] = None, target: Optional[list] = None,
                 mode: int = 0, feature_map: Optional[MnemonicDictionary]= None, config: dict = None):
        self._parser = Parser()
        self._data: Optional[welly.Project] = None
        self._scaler = None
        self._folds: Optional[dict] = None
        self._zen_angle = {}
        self._data_name = data_name
        self._features_lst = features
        self._target_lst = target
        self.mode = mode
        self._feature_map = feature_map
        self._create_features = config['create_features']
        self._normalize = config['normalize']
        self._fields = config['fields']
        self._measure_freq = config['measure_freq_m']
        self._clip_by_limits = config['clip_by_limits']
        self._cut_beginning_m = config['cut_beginning_m']
        self._remove_dupl = config['remove_duplicates_wells']
        self._remove_shoe = config['remove_shoe']
        self._max_shoe_size_m = config['max_shoe_size_m']
        self._filter_horizont = config['filter_horizont']
        self._exclude_wells = config['exclude_wells']

        self._load_data()
        self._load_zenit_angle()

    def _load_data(self):
        self._data = self._parser.load(self._data_name)
        return self._data

    def _create_new_features(self, well):
        new_mn_lst = list()
        mn_list = ['GR', 'RES_SH_PH', 'RES_MED_PH', 'RES_D_PH']
        for mn in self._feature_map.get_mnemonics(self._features_lst, well).keys():
            if mn in self._feature_map.targets:
                continue
            # if mn not in mn_list:
            #     continue

            for lag_size in [1, 2, 3, 4]:
                lag_mn = f'{mn}_lag_{lag_size}'
                diff_mn = f'{mn}_diff_{lag_size}'
                # new_mn_lst.append(lag_mn)
                new_mn_lst.append(diff_mn)
                lag_df = well[mn].to_frame().rename(columns={mn: lag_mn})
                lag_df = lag_df.shift(lag_size)
                diff_df = well[mn].to_frame().rename(columns={mn: diff_mn})
                diff_df[diff_mn] = lag_df[lag_mn] - diff_df[diff_mn]
                # well = pd.merge(well, lag_df, left_index=True, right_index=True)
                well = pd.merge(well, diff_df, left_index=True, right_index=True)
            for roll_size in [4, 20]:
                stats_mn = [f'{mn}_{roll_size}_min', f'{mn}_{roll_size}_max']
                window = well[mn].rolling(roll_size)
                wind_stats = pd.concat([window.min(), window.max()], axis=1)
                wind_stats.columns = stats_mn
                new_mn_lst.extend(stats_mn)
                well = pd.merge(well, wind_stats, left_index=True, right_index=True)

        return well.dropna(), new_mn_lst

    @staticmethod
    def normalize_df(data_df: pd.DataFrame, mode: int) -> pd.DataFrame:
        scaler = StandardScaler()
        scaler = scaler.fit(data_df)
        data_sc = scaler.transform(data_df)
        data_df_sc = pd.DataFrame(data=data_sc, index=data_df.index, columns=data_df.columns)
        Preprocess.save_scaler(scaler, file_path=os.path.join(INDICES_DIR, str(mode), 'scaler.npy'))
        return data_df_sc

    @staticmethod
    def save_scaler(scaler, file_path: str = None):
        if not file_path:
            file_path = os.path.join(INDICES_DIR, 'scaler.npy')
        with open(file_path, 'wb') as pkl:
            pickle.dump(scaler, pkl)
        task = Task.current_task()
        if task:
            task.upload_artifact('Scaler', file_path)

    @staticmethod
    def load_scaler(path: str = None):
        if not path:
            path = os.path.join(INDICES_DIR, 'scaler.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, 'rb') as pkl:
            obj = pickle.load(pkl)
        return obj

    def process(self):
        self._folds = dict()
        not_used_wells_count = 0

        for well in self._data:
            if well.uwi in self._exclude_wells:
                not_used_wells_count += 1
                continue
            field, well_name = well.uwi.split('_', maxsplit=1)

            try:
                latitude = float(re.sub('[^\d\.]', '', str(well.location.latitude)))
                longitude = float(re.sub('[^\d\.]', '', str(well.location.longitude)))
            except (AttributeError, ValueError):
                latitude, longitude = 1.0, 1.0

            if self._fields and field not in self._fields:
                continue

            if self._remove_dupl and '_DUPL' in well_name:
                not_used_wells_count += 1
                continue

            fold = well.df(basis=well.survey_basis(step=self._measure_freq))

            # use BotuH Layer info to filter wells
            # if 'BTH' not in fold.columns.values:
            #     continue

            # filter by zenit angle
            # fold = self._filter_data(fold, field, well_name, filter_horizont=self._filter_horizont)

            if self._check_is_fold_none(fold):
                not_used_wells_count += 1
                continue

            # get list of mnemonics for specific set of features
            mnemonics_list = self._feature_map.get_mnemonics(self._features_lst, fold)

            if not mnemonics_list:
                not_used_wells_count += 1
                continue

            # filter by mnemonic names
            fold = self.process_nan(fold[list(mnemonics_list.keys())])
            # fold = self.process_nan(fold[list(mnemonics_list.keys()) + ['BTH']])
            fold = fold.rename(columns=mnemonics_list)
            fold.attrs['original_mnemonics'] = mnemonics_list

            if self._check_is_fold_none(fold):
                not_used_wells_count += 1
                continue

            if self._clip_by_limits:
                fold = self._clip_limits(fold)

            # drop shoe part of well
            if self._remove_shoe:
                fold = self.drop_shoe(fold)

            # cut beginning of well
            if self._cut_beginning_m:
                fold = self._cut_beginning(fold, self._cut_beginning_m)

            if self._check_is_fold_none(fold):
                not_used_wells_count += 1
                continue

            # filter by limit RHOB
            # if not self._filter_rhob(fold):
            #     continue

            # filter by depth
            depth_idx = fold.axes[0].values
            if not self._check_depth_len(depth_idx.tolist()):
                not_used_wells_count += 1
                continue

            # filter by iqr
            mask = self._filter_by_iqr(fold, self._feature_map)
            # mask = pd.DataFrame(True, index=fold.index, columns=fold.columns)

            if self._create_features:
                fold, new_feat_lst = self._create_new_features(fold)
                new_feat_mask = pd.DataFrame(True, index=fold.index, columns=new_feat_lst)
                mask = pd.merge(mask, new_feat_mask, left_index=True, right_index=True).dropna()

            depth_idx = fold.axes[0].values
            uwi_idx = ['_'.join([field, well_name])] * len(fold)

            tuples = list(zip(*[uwi_idx, depth_idx]))
            index = pd.MultiIndex.from_tuples(tuples, names=["UWI", "DEPT"])
            fold = fold.set_index(index)
            mask = mask.set_index(index)

            # botuh = fold[['BTH']].astype(bool)
            # fold = fold.drop(columns=['BTH'], axis=1)
            botuh = None

            self._folds.update({well_name: {'data': fold, 'FIELD': field,
                                      'LATI': latitude, 'LONG': longitude, 'mask': mask, 'botuh': botuh}})
        print(f'Used wells: {len(self._folds)}')
        print(f'Not used wells: {not_used_wells_count}')

    def save_data(self):
        # remove output dir first if exists
        output_dir = os.path.join(INDICES_DIR, str(self.mode))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        self._parser.save_obj(self._folds, os.path.join(output_dir, 'data'),
                              data_name='Dataset file')

    def drop_shoe(self, df: pd.DataFrame, mnems: List[str] = None):
        """
        Drop shoe of well. Shoe is a part of well with casing and cement. It is not a part of reservoir.
        It's assumed to use RES_SH_PH and RES_D_PH mnemonics to find shoe of well.
        Args:
            df: dataframe with data
            mnems: list of mnemonics to use for shoe detection

        Returns:
            dataframe without shoe
        """
        if not mnems:
            mnems = ['RES_SH_PH', 'RES_D_PH']

        cond_ind = None
        max_val_probably_shoe_last_ind = 0
        for mn in mnems:
            mn_cond_ind = abs(df[mn].diff().rolling(15).mean()) > 10
            max_mn_val = df[mn].max()
            max_val_gaps = df[df[mn]==max_mn_val].index.to_series().diff().dropna()
            if max_val_gaps.size == 0:
                max_val_gaps_lt = mn_cond_ind
            else:
                max_val_gaps_lt = max_val_gaps.lt(20)
                # only true, no false. all gaps < 20, so we can crop from last max val
                if max_val_gaps_lt.min() == True:
                    last_max_mn_val_bef_gap_ind = df[mn][::-1].idxmax()
                else:
                    first_big_gap_end_ind = max_val_gaps_lt.idxmin()
                    gap_val = max_val_gaps[max_val_gaps_lt.idxmin()]
                    last_max_mn_val_bef_gap_ind = first_big_gap_end_ind - gap_val
                max_val_gaps_lt.values[:] = 0
                if last_max_mn_val_bef_gap_ind - df[mn].index[0] < self._max_shoe_size_m:
                    max_val_gaps_lt.loc[last_max_mn_val_bef_gap_ind] = True
                    if max_val_probably_shoe_last_ind < last_max_mn_val_bef_gap_ind:
                        max_val_probably_shoe_last_ind = last_max_mn_val_bef_gap_ind

            cond_ind = mn_cond_ind | max_val_gaps_lt if cond_ind is None else (cond_ind & mn_cond_ind) \
                                                            | max_val_gaps_lt
        # cond_ind[::-1].idxmax() - index of last shoe point
        shoe_end_ind = cond_ind[::-1].idxmax() + 3 if  cond_ind[::-1].idxmax() != df.index[-1] else df.index[0]
        shoe_end_ind = max(shoe_end_ind, max_val_probably_shoe_last_ind+3)
        if shoe_end_ind - df.index[0] > self._max_shoe_size_m:
            shoe_end_ind = df.index[0]

        return df.loc[shoe_end_ind:]

    @staticmethod
    def process_nan(df: pd.DataFrame, strategy: str = 'drop'):
        if strategy == 'drop':
            df = df.dropna()
        else:
            print(f'Unknown processing nans strategy {strategy}!')
        return df

    def _filter_data(self, df: pd.DataFrame, field: str,  well_name: str, filter_horizont: bool = True) \
            -> Union[pd.DataFrame, None]:

        # horizont minimum
        zen_angles = self._zen_angle[field][well_name]
        if filter_horizont:
            end = zen_angles['DEPTH'] > max(zen_angles['DEPTH']) - 500
            zen = zen_angles['ZENIT'][end]
            angle = np.mean(zen)
            if angle < 80:
                return None

        goal_depth = min(zen_angles['DEPTH'][zen_angles['ZENIT'] > min_zenit_angle])
        df = df[df.index >= goal_depth]
        return df

    def _clip_limits(self, df: pd.DataFrame, mnemonics: Optional[list] = None) -> pd.DataFrame:
        """
        Clip data using limits from feature map. If mnemonics is None, then clip all features.
        If there is no limits for feature, then do not clip it.
        Args:
            df: dataframe with data
            mnemonics: list of mnemonics to clip. If None, clip all mnemonics

        Returns:
            clipped dataframe
        """

        if mnemonics is None:
            mnemonics = self._feature_map.get_mnemonics(self._features_lst, df).keys()

        for mn in mnemonics:
            limits = self._feature_map.get_limits(mn)
            if limits:
                df[mn] = df[mn].clip(limits[0], limits[1])
        return df

    @staticmethod
    def _cut_beginning(df: pd.DataFrame, val_to_cut_m: int = 0) -> pd.DataFrame:
        """
        Cut beginning of dataframe
        Args:
            df: dataframe to cut
            val_to_cut_m: value to cut from beginning in meters

        Returns:
            cut dataframe
        """
        # cut all data before start depth + val_to_cut_m
        df = df[df.index >= df.index[0] + val_to_cut_m]
        return df

    def _load_zenit_angle(self):
        for filename in os.listdir(os.path.join(DATA_PATH, 'utils')):
            if filename.startswith('zen_angle'):
                with open(os.path.join(DATA_PATH, 'utils', filename), 'rb') as f:
                    self._zen_angle[filename.split('_')[-1][:-4]] = pickle.load(f)

    @staticmethod
    def _check_depth_len(depth_values: list, permissible_value: int = 50) -> bool:
        depth_len = depth_values[-1] - depth_values[0]
        return depth_len > permissible_value

    @staticmethod
    def _filter_by_iqr(data, feature_map) -> pd.DataFrame:
        res = pd.DataFrame(index=data.index.values, columns=data.columns.values, dtype=bool)

        for col in data.columns:
            if col == 'RHOB':
                mask = data[col].between(1.85, 2.95)
            elif col in feature_map.targets:
                continue
            elif col == 'BTH':
                mask = data[col].astype(bool)
            elif 'PH' in col:
                continue
            elif 'ATT' in col:
                mask = data[col].between(0, 200)
            else:
                mask, _ = feature_map.filter_by_iqr(data, col)

            res[col] = mask

        return res

    @staticmethod
    def _check_is_fold_none(fold):
        return fold is None or fold.shape[0] == 0

    def use_filter_mask(self, data_df: pd.DataFrame, mask: pd.DataFrame):
        # set NaN if data is out of limits
        data_df = data_df.mask(~mask)
        data_df = self.process_nan(data_df)

        return data_df

    @staticmethod
    def _filter_rhob(data: pd.DataFrame):
        hist, bins = np.histogram(data[['RHOB']].to_numpy(), bins=100)
        lim = bins[hist.argmax()]
        if lim < 2.3:
            return False
        return True

    @staticmethod
    def _filter_noise_well(data, feature_map) -> bool:
        for col in data.columns:
            if col in feature_map.targets or col == 'BTH':
                continue
            else:
                if feature_map.filter_noise(data, col):
                    return True
        return False


def inv_transform(scaler, data: np.ndarray, col_name: Union[str, List[str]], col_names: List[str]) -> np.ndarray:
    """
    Inverse transform for normalized data array using scaler object

    Args:
        scaler (object): Scaler object, for example StandardScaler
        data (np.ndarray): Array with data to inverse normalize operation
        col_name (str or List[str]): Column name/names from data
        col_names (List[str]): All columns names in save order that was used at scaler.transform operation

    Returns:
        np.ndarray: array with denormalized data with same as `data` shape
    """
    dummy = pd.DataFrame(np.zeros((len(data), len(col_names))), columns=col_names)
    dummy[col_name] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=col_names)
    return dummy[col_name].values
