import lzma
import os
import pickle
from typing import List, Any

from clearml import Task

from src.parsing.las import load_project_from_dir
from src.config import DATA_PATH, ARTIFACTS_DIR, SOURCE_DATA_DIR, silent_mode


class Parser:
    def __init__(self, db_type: str = 'pickle'):
        self._parsed_data = None
        self._artifacts_dir = ARTIFACTS_DIR
        self._source_dir = SOURCE_DATA_DIR
        self._data_dir = DATA_PATH
        self._db_type = db_type
        self._out_db_name = 'data'

    def parse(self, formats: List[str], data_dir: str = None, **kwargs):
        source_path = data_dir if data_dir is not None else self._source_dir

        if 'las' in formats:
            self._parsed_data = load_project_from_dir(source_path, use_cache=False, **kwargs)

    def load(self, file_name: str):
        file_path = os.path.join(self._data_dir, file_name)
        return self.load_obj(file_path, self._db_type)

    def save(self, out_dir: str = None, out_name: str = None):
        out_path = os.path.join(self._data_dir if out_dir is None else out_dir,
                                self._out_db_name if out_name is None else out_name)
        self.save_obj(self._parsed_data, out_path, self._db_type)

    @staticmethod
    def load_obj(file_path: str, db_type: str = 'pickle'):
        if db_type == 'pickle':
            file_path += '.pkl'
            if not os.path.exists(file_path):
                if not silent_mode:
                        print(f'Cache file does not exist at {file_path}')
                return None
            with lzma.open(file_path) as f:
                data = pickle.load(f)
            return data
        else:
            raise Exception(f'Can not load parsed data with {db_type} storage type!')

    @staticmethod
    def save_obj(data: Any, out_path: str, db_type: str = 'pickle', data_name: str = None):
        task = Task.current_task()
        if db_type == 'pickle':
            with lzma.open(out_path + '.pkl', "wb") as f:
                pickle.dump(data, f)
            if task is not None and data_name is not None:
                task.upload_artifact(data_name, out_path + '.pkl')
        else:
            raise Exception(f'Can not save parsed data with {db_type} storage type!')
