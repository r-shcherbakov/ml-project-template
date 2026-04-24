# Explicit Pipeline Step Parameters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `params.yaml` with typed Pydantic `BaseModel` classes injected as required constructor arguments into each pipeline step.

**Architecture:** Each step that needs parameters defines a `XxxParams(BaseModel)` in its own file; the step constructor takes `params: XxxParams` as a required argument (no default). `BasePipelineStep` accepts `params: Optional[BaseModel] = None` and calls `task.connect(params.model_dump())` to preserve ClearML UI visibility. `pipeline.py` passes params explicitly at every call site.

**Tech Stack:** Pydantic v2 (`BaseModel`), ClearML (`task.connect`), Python 3.9+

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/core/pipeline_step.py` | Modify | Replace `_init_parameters()` with `_connect_params()`; accept `params` in `__init__` |
| `src/preprocess/preprocess_pipeline_step.py` | Modify | Add `PreprocessParams`; make `params` required |
| `src/features/split_dataset_pipeline_step.py` | Modify | Add `SplitDatasetParams`; replace dict mutations with instance vars |
| `src/train/train_pipeline_step.py` | Modify | Add `TrainParams`; fix `_set_ignored_features`, `_run_cv`, `_get_prediction` |
| `src/pipelines/pipeline.py` | Modify | Import Params models; pass explicitly at every step call |
| `src/settings.py.jinja` | Modify | Remove `params_path` field and `FilePath` import |
| `src/params.yaml` | Delete | No longer needed |
| `tests/test_step_params.py` | Create | Unit tests for all Params models (runs in generated projects) |

> **Note:** `src/features/feature_engineer_pipeline_step.py` and `src/visualization/plotting_pipeline_step.py` use no `self.step_params` — no changes needed. `select_features` step has no step class — ignore.

> **Test context:** Tests live in `copier-python-template/tests/` and become part of every generated project. They use `unittest.mock` to avoid a live ClearML connection.

---

### Task 1: Write failing tests for Params models

**Files:**
- Create: `copier-python-template/tests/test_step_params.py`

- [ ] **Step 1: Write failing tests**

```python
# copier-python-template/tests/test_step_params.py
"""Unit tests for pipeline step Params models and BasePipelineStep._connect_params."""
from typing import Optional, List
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel, ValidationError


# ── Params model stubs (will fail import until implemented) ──────────────────
def test_preprocess_params_defaults():
    from src.preprocess.preprocess_pipeline_step import PreprocessParams
    p = PreprocessParams()
    assert p.skip_mark is False


def test_preprocess_params_override():
    from src.preprocess.preprocess_pipeline_step import PreprocessParams
    p = PreprocessParams(skip_mark=True)
    assert p.skip_mark is True


def test_split_dataset_params_defaults():
    from src.features.split_dataset_pipeline_step import SplitDatasetParams
    p = SplitDatasetParams()
    assert p.split_test is True
    assert p.num_test_objects == 2
    assert p.test_objects is None


def test_train_params_defaults():
    from src.train.train_pipeline_step import TrainParams
    p = TrainParams()
    assert p.skip_cv is True
    assert p.n_splits == 4
    assert p.train_final_model is True
    assert p.binary_threshold == 0.5


def test_train_params_type_validation():
    from src.train.train_pipeline_step import TrainParams
    with pytest.raises(ValidationError):
        TrainParams(n_splits="not_an_int")


def test_preprocess_step_requires_params():
    """Calling PreprocessPipelineStep() without params raises TypeError."""
    from src.preprocess.preprocess_pipeline_step import PreprocessPipelineStep
    with pytest.raises(TypeError):
        PreprocessPipelineStep()


def test_split_dataset_step_requires_params():
    from src.features.split_dataset_pipeline_step import SplitDatasetPipelineStep
    with pytest.raises(TypeError):
        SplitDatasetPipelineStep()


def test_train_step_requires_params():
    from src.train.train_pipeline_step import TrainPipelineStep
    with pytest.raises(TypeError):
        TrainPipelineStep()
```

- [ ] **Step 2: Confirm tests fail (import errors expected at this stage)**

```
These tests currently fail with ImportError (PreprocessParams not defined yet).
That is the expected RED state — proceed to Task 2.
```

---

### Task 2: Update `BasePipelineStep`

**Files:**
- Modify: `src/core/pipeline_step.py`

- [ ] **Step 1: Replace `_init_parameters` with `_connect_params`, update `__init__`**

Replace the entire file content. Key changes:
- Remove `import yaml`
- Add `from pydantic import BaseModel` 
- Add `params: Optional[BaseModel] = None` parameter to `__init__`
- Store `self.step_params = params`
- Replace `self._init_parameters()` call with `self._connect_params()`
- Remove `_init_parameters` method
- Add `_connect_params` method

```python
# -*- coding: utf-8 -*-
""" Base pipeline step """
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import traceback
from typing import Any, Optional, Union, TYPE_CHECKING

from clearml import Task, Dataset
import pandas as pd
from pydantic import BaseModel

from src.common.exceptions import (
    DatasetDownloadError,
    PipelineExecutionError,
)
from src.utilities.utils import compress_pickle
from src.utilities.path_utils import is_empty_dir
from src.settings import SETTINGS

if TYPE_CHECKING:
    from src.common.pipeline_steps import PipelineStep
    from src.settings import Settings


class BasePipelineStep(ABC):
    r"""Abstract class for all pipeline steps."""

    def __init__(
        self,
        pipeline_step: 'PipelineStep',
        params: Optional[BaseModel] = None,
    ):
        self.pipeline_step: 'PipelineStep' = pipeline_step
        self.settings: 'Settings' = SETTINGS
        self.step_params = params

        self._init_task()
        self._connect_params()

    def _init_task(self):
        self.task: Task = Task.init(
            project_name=self.settings.clearml.project,
            task_name=f'{self.pipeline_step.name} task',
            task_type=self.pipeline_step.task_type,
            tags=self.settings.clearml.tags,
            deferred_init=True,
            reuse_last_task_id=False)
        if self.settings.clearml.execute_remotely:
            self.task.execute_remotely(queue_name=self.settings.clearml.queue_name)

    def _connect_params(self):
        if self.step_params is not None:
            self.task.connect(
                self.step_params.model_dump(),
                name=self.pipeline_step.name.replace('_', ' '),
            )

    def _log_success_step_execution(
        self,
        file_name: str,
    ) -> None:
        self.task.logger.report_text(
            f"Execution {self.pipeline_step.name.replace('_', ' ')} for {file_name} successfully finished",
            level=logging.INFO
        )

    def _log_failed_step_execution(
        self,
        file_name: str,
        exception: Exception,
    ) -> None:
        self.task.logger.report_text(
            f"Execution {self.pipeline_step.name.replace('_', ' ')} for {file_name} failed due to: {exception}",
            level=logging.INFO
        )
        self.task.logger.report_text(
            'traceback:' + traceback.format_exc(),
            level=logging.DEBUG,
            print_console=False,
        )

    def _log_success_upload_dataset(self) -> None:
        self.task.logger.report_text(
            f"Dataset for {self.pipeline_step.name.replace('_', ' ')} step "
            f"successfully uploaded",
            level=logging.INFO
        )

    def _log_failed_upload_dataset(
        self,
        exception: Exception,
    ) -> None:
        self.task.logger.report_text(
            f"Uploading dataset for {self.pipeline_step.name.replace('_', ' ')} step "
            f"failed due to: {exception}",
            level=logging.INFO
        )
        self.task.logger.report_text(
            'traceback:' + traceback.format_exc(),
            level=logging.DEBUG,
            print_console=False,
        )

    def _log_success_save_data(
        self,
        file_name: str,
    ) -> None:
        self.task.logger.report_text(
            f"Data of {self.pipeline_step.name.replace('_', ' ')} step "
            f"for {file_name} successfully locally saved",
            level=logging.INFO
        )

    def _log_failed_save_data(
        self,
        file_name: str,
        exception: Exception,
    ) -> None:
        self.task.logger.report_text(
            f"Savings data of {self.pipeline_step.name.replace('_', ' ')} step "
            f"for {file_name} failed due to: {exception}",
            level=logging.INFO
        )
        self.task.logger.report_text(
            'traceback:' + traceback.format_exc(),
            level=logging.DEBUG,
            print_console=False,
        )

    def _download_input_dataset(self, dataset_id: str) -> None:
        remote_dataset = Dataset.get(
            dataset_id=dataset_id,
            only_completed=True,
        )

        try:
            _ = remote_dataset.get_mutable_local_copy(self._input_directory)
        except ValueError:
            raise DatasetDownloadError()


    def _upload_output_dataset(self, parent_datasets: list[str]) -> str:
        dataset = Dataset.create(
            dataset_project=self.settings.clearml.project,
            dataset_name=f"{self.pipeline_step.name.replace('_', ' ')} dataset",
            dataset_tags=self.settings.clearml.tags,
            parent_datasets=parent_datasets,
        )
        try:
            dataset.add_files(path=self._output_directory)
            dataset.finalize(auto_upload=True)
            self._log_success_upload_dataset()
            return dataset.id
        except Exception as exception:
            self._log_failed_upload_dataset(
                exception=exception,
            )
            raise PipelineExecutionError

    def _save_locally_data(
        self,
        path: Union[str, Path],
        data: pd.DataFrame,
    ) -> None:
        try:
            file_name = compress_pickle(path, data).stem
            self._log_success_save_data(file_name=file_name)
        except Exception as exception:
            self._log_failed_save_data(
                file_name=file_name,
                exception=exception,
            )

    @property
    def _input_directory(self):
        return self.pipeline_step.input_directory

    @property
    def _output_directory(self):
        return self.pipeline_step.output_directory

    @abstractmethod
    def start(self):
        pass
```

- [ ] **Step 2: Commit**

```bash
git add copier-python-template/src/core/pipeline_step.py \
        copier-python-template/tests/test_step_params.py
git commit -m "refactor: replace _init_parameters with _connect_params in BasePipelineStep"
```

---

### Task 3: Update `PreprocessPipelineStep`

**Files:**
- Modify: `src/preprocess/preprocess_pipeline_step.py`

- [ ] **Step 1: Add `PreprocessParams`, update `__init__`, replace `.get()` with attribute access**

```python
# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
import logging
from pathlib import Path
from typing import List, Union
import warnings

from pydantic import BaseModel
from sklearn import set_config
from sklearn.pipeline import Pipeline

from src.common.constants import GENERAL_EXTENSION
from src.common.pipeline_steps import PREPROCESS
from src.core import BasePipelineStep
from src.utilities.loaders import CsvLoader
from src.utilities.path_utils import is_empty_dir
from src.preprocess.preprocessor import Preprocessor, MarkDataTransformer

warnings.simplefilter(action="ignore", category=FutureWarning)


class PreprocessParams(BaseModel):
    skip_mark: bool = False


class PreprocessPipelineStep(BasePipelineStep):
    def __init__(self, params: PreprocessParams):
        super().__init__(PREPROCESS, params=params)

    @property
    def _input_files(self) -> List[Path]:
        input_directory = self._input_directory
        file_type = r"/*csv"
        input_filepath_files = [
            Path(file_path) for file_path in glob(str(input_directory) + file_type)
        ]
        return input_filepath_files

    def _transform_input_data(self, file_path: Union[Path, str]) -> None:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        file_name = Path(file_path).stem.replace(" ", "").upper()
        self.task.logger.report_text(
            f"Processing of {file_name}",
            level=logging.DEBUG,
            print_console=False,
        )

        data = CsvLoader(path=file_path).load()
        if self.step_params.skip_mark:
            step_pipeline = Pipeline(
                [
                    ("preprocessor", Preprocessor())
                 ]
            )
        else:
            step_pipeline = Pipeline(
                steps=[
                    ("preprocessor", Preprocessor()),
                    ("add_target", MarkDataTransformer()),
                 ]
            )
        set_config(transform_output="pandas")

        try:
            preprocessed = step_pipeline.transform(data)
            self._log_success_step_execution(file_name=file_name)
        except Exception as exception:
            self._log_failed_step_execution(
                file_name=file_name,
                exception=exception,
            )

        preprocessed_filepath = Path(
            os.path.join(
                self._output_directory, f"{file_name}{GENERAL_EXTENSION}"
            )
        )
        self._save_locally_data(
            path=preprocessed_filepath,
            data=preprocessed,
        )

        del preprocessed, data
        gc.collect()

    def start(self, dataset_id: str) -> str:
        if is_empty_dir(self._input_directory):
            self._download_input_dataset(dataset_id=dataset_id)

        for path in self._input_files:
            self._transform_input_data(path)

        return self._upload_output_dataset(parent_datasets=[dataset_id])
```

- [ ] **Step 2: Commit**

```bash
git add copier-python-template/src/preprocess/preprocess_pipeline_step.py
git commit -m "refactor: add PreprocessParams, make params required in PreprocessPipelineStep"
```

---

### Task 4: Update `SplitDatasetPipelineStep`

**Files:**
- Modify: `src/features/split_dataset_pipeline_step.py`

- [ ] **Step 1: Add `SplitDatasetParams`, update `__init__`, replace dict mutations with instance var**

Key changes:
- `self.step_params.get('split_test', False)` → `self.step_params.split_test`
- `self.step_params.get("test_objects", None)` → `self.step_params.test_objects`
- `self.step_params.get("num_test_objects", 1)` → `self.step_params.num_test_objects`
- `self.step_params["test_objects"] = self.test_objects` → removed (already stored as `self.test_objects`)

```python
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
from pydantic import BaseModel
from tqdm import tqdm

from src.common.constants import GENERAL_EXTENSION
from src.common.exceptions import PipelineExecutionError
from src.common.features import GROUP_ID
from src.common.pipeline_steps import SPLIT_DATASET
from src.core import BasePipelineStep
from src.utilities.loaders import PickleLoader
from src.utilities.path_utils import is_empty_dir

warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitDatasetParams(BaseModel):
    split_test: bool = True
    num_test_objects: int = 2
    test_objects: Optional[List[str]] = None


class SplitDatasetPipelineStep(BasePipelineStep):
    def __init__(self, params: SplitDatasetParams):
        super().__init__(SPLIT_DATASET, params=params)

    @property
    def _input_files(self) -> List[Path]:
        self._check_input_directory()
        input_directory = self._input_directory
        file_type = f"/*{GENERAL_EXTENSION}"
        input_filepath_files = [
            Path(file_path) for file_path in glob(str(input_directory) + file_type)
        ]
        return input_filepath_files

    def _set_test_objects(self) -> None:
        if self.step_params.split_test:
            if self.step_params.test_objects:
                self.test_objects: List[str] = self.step_params.test_objects
            else:
                self.test_objects = [
                    Path(file_path).stem \
                    for file_path in random.sample(
                        self._input_files, self.step_params.num_test_objects
                    )
                ]
        else:
            self.test_objects = []

    def _log_groups_mapping(self) -> None:
        self.file_name_mapping: Dict[str, int] = {
            Path(file_path).stem.replace(" ", "").upper(): number \
                for number, file_path in enumerate(self._input_files)
        }
        self.task.upload_artifact("group_mapping", self.file_name_mapping)

    def _concatenate_dataframes(self) -> None:
        train = pd.DataFrame()
        test = pd.DataFrame()
        try:
            for file_path in tqdm(self._input_files, total=len(self._input_files)):
                file_name = Path(file_path).stem
                self.task.logger.report_text(
                    f"Processing of {file_name}",
                    level=logging.DEBUG,
                    print_console=False,
                )
                data = PickleLoader(path=file_path).load()
                data[GROUP_ID.name] = self.file_name_mapping[file_name]

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
            path=Path(os.path.join(self._output_directory, f"train{GENERAL_EXTENSION}")),
            data=train,
        )

        if self.step_params.split_test and not test.empty:
            self._save_locally_data(
                path=Path(os.path.join(self._output_directory, f"test{GENERAL_EXTENSION}")),
                data=test,
            )

    def start(self, dataset_id: str) -> str:
        if is_empty_dir(self._input_directory):
            self._download_input_dataset(dataset_id=dataset_id)

        self._set_test_objects()
        self._log_groups_mapping()
        self._concatenate_dataframes()

        return self._upload_output_dataset(parent_datasets=[dataset_id])
```

- [ ] **Step 2: Commit**

```bash
git add copier-python-template/src/features/split_dataset_pipeline_step.py
git commit -m "refactor: add SplitDatasetParams, replace dict mutations with instance vars"
```

---

### Task 5: Update `TrainPipelineStep`

**Files:**
- Modify: `src/train/train_pipeline_step.py`

- [ ] **Step 1: Add `TrainParams`, update `__init__`, fix all `self.step_params` usages**

Key changes:
- `self.step_params.pop("n_splits", 2)` → `self.step_params.n_splits`
- `self.step_params["ignored_features"] = ignored_features` → `self.ignored_features = ignored_features` (instance var)
- `CatBoostClassifier(**self.step_params, ...)` → `CatBoostClassifier(random_seed=self.settings.random_seed)` (step_params dict-spreading to CatBoost was incorrect — `skip_cv`/`train_final_model` are not CatBoost hyperparams)
- `self.step_params.get("binary_threshold", 0.5)` → `self.step_params.binary_threshold`

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add copier-python-template/src/train/train_pipeline_step.py
git commit -m "refactor: add TrainParams, replace dict access with typed attribute access"
```

---

### Task 6: Update `pipeline.py`

**Files:**
- Modify: `src/pipelines/pipeline.py`

- [ ] **Step 1: Import Params models and pass params explicitly at every step call**

```python
import os
from pathlib import Path
from typing import Tuple, TYPE_CHECKING

from clearml import PipelineController, Dataset
import pandas as pd

from src.common.exceptions import PipelineExecutionError
from src.common.pipeline_steps import (
    PRERUN,
    PREPROCESS,
    FEATURE_ENGINEER,
    SPLIT_DATASET,
    TRAIN,
)
from src.features import (
    FeatureEngineerPipelineStep,
    SplitDatasetPipelineStep,
)
from src.features.split_dataset_pipeline_step import SplitDatasetParams
from src.preprocess import PreprocessPipelineStep
from src.preprocess.preprocess_pipeline_step import PreprocessParams
from src.train import TrainPipelineStep
from src.train.train_pipeline_step import TrainParams
from src.settings import SETTINGS
from src.utilities.path_utils import is_empty_dir

if TYPE_CHECKING:
    from src.features.feature_engineer import FeatureEngineer


def run_prerun_step() -> str:
    if not is_empty_dir(SETTINGS.storage.raw_folder):
        try:
            # Upload raw data from remote storage
            pass
        except Exception:
            raise PipelineExecutionError("Raw data is not available")

        remote_dataset = Dataset.create(
            dataset_project=SETTINGS.clearml.project,
            dataset_name="raw data",
            dataset_tags=SETTINGS.clearml.tags,
        )
        remote_dataset.add_files(path=SETTINGS.storage.raw_folder)
        remote_dataset.finalize(auto_upload=True)

        return remote_dataset.id


def run_preprocess_step(dataset_id: str) -> str:
    return PreprocessPipelineStep(
        params=PreprocessParams(skip_mark=False),
    ).start(dataset_id=dataset_id)


def run_split_dataset_step(dataset_id: str) -> str:
    return SplitDatasetPipelineStep(
        params=SplitDatasetParams(
            split_test=True,
            num_test_objects=2,
        ),
    ).start(dataset_id=dataset_id)


def run_feature_engineer_step(dataset_id: str) -> Tuple['FeatureEngineer', str]:
    return FeatureEngineerPipelineStep.start(dataset_id=dataset_id)


def run_train_step(dataset_id: str) -> None:
    return TrainPipelineStep(
        params=TrainParams(
            skip_cv=True,
            n_splits=4,
            train_final_model=True,
            binary_threshold=0.5,
        ),
    ).start(dataset_id=dataset_id)


if __name__ == '__main__':

    pipe = PipelineController(
        name=f'{SETTINGS.clearml.project} pipeline',
        project=SETTINGS.clearml.project,
        target_project=SETTINGS.clearml.project,
        add_pipeline_tags=False,
        auto_version_bump=True,
        add_run_number=False,
        packages="./requirements.txt",
    )

    pipe.add_function_step(
        name=PRERUN.name,
        task_type=PRERUN.task_type,
        function=run_prerun_step,
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=PREPROCESS.name,
        task_type=PREPROCESS.task_type,
        parents=[PRERUN.name],
        function=run_preprocess_step,
        function_kwargs=dict(
            dataset_id='${prerun.dataset_id}'
        ),
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=SPLIT_DATASET.name,
        task_type=SPLIT_DATASET.task_type,
        parents=[PREPROCESS.name],
        function=run_split_dataset_step,
        function_kwargs=dict(
            dataset_id='${preprocess.dataset_id}'
        ),
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=FEATURE_ENGINEER.name,
        task_type=FEATURE_ENGINEER.task_type,
        parents=[SPLIT_DATASET.name],
        function=run_feature_engineer_step,
        function_kwargs=dict(
            dataset_id='${split_dataset.dataset_id}',
        ),
        function_return=['feature_engineer', 'dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.add_function_step(
        name=TRAIN.name,
        task_type=TRAIN.task_type,
        parents=[FEATURE_ENGINEER.name],
        function=run_train_step,
        function_kwargs=dict(
            dataset_id='${feature_engineer.dataset_id}',
        ),
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
        time_limit=SETTINGS.clearml.time_limit,
    )

    pipe.set_default_execution_queue(SETTINGS.clearml.queue_name)
    if SETTINGS.clearml.execute_remotely:
        pipe.start(queue=SETTINGS.clearml.queue_name)
    else:
        pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline finished")
```

- [ ] **Step 2: Commit**

```bash
git add copier-python-template/src/pipelines/pipeline.py
git commit -m "refactor: pass params explicitly at every pipeline step call site"
```

---

### Task 7: Remove `params.yaml` and `params_path` from settings

**Files:**
- Delete: `src/params.yaml`
- Modify: `src/settings.py.jinja`

- [ ] **Step 1: Delete `params.yaml`**

```bash
git rm copier-python-template/src/params.yaml
```

- [ ] **Step 2: Remove `params_path` from `settings.py.jinja`**

In `src/settings.py.jinja`, remove the `params_path` field from `Settings` and `FilePath` from the pydantic imports:

Remove from pydantic imports:
```python
# Before:
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    Field,
    computed_field,
    field_validator
)

# After:
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    computed_field,
    field_validator
)
```

Remove from `Settings` class:
```python
# Remove this field entirely:
params_path: FilePath = Field(
    os.path.join(Path(__file__).resolve().parent, 'params.yaml'),
    description='Path to the experiment parameters config'
)
```

- [ ] **Step 3: Commit**

```bash
git add copier-python-template/src/settings.py.jinja
git commit -m "refactor: remove params_path from Settings, delete params.yaml"
```

---

### Task 8: Update knowledge base

**Files:**
- Modify: `docs/knowledge/generated-project/params-yaml-contract.md`

- [ ] **Step 1: Replace the params-yaml-contract doc with the new pattern**

Replace the entire file with:

```markdown
# Pipeline Step Parameters Contract

## Context

Each pipeline step that has configurable parameters defines a `XxxParams(BaseModel)`
class in the **same file** as the step class. Parameters are passed as a **required**
constructor argument — there is no default. `BasePipelineStep._connect_params()` calls
`task.connect(params.model_dump())` to make parameters visible and editable in the
ClearML UI without code changes.

## Invariants

1. `XxxParams` inherits from `pydantic.BaseModel` and lives in the same file as its step.
2. `params` is a **required** constructor argument — `XxxStep()` with no args raises `TypeError`.
3. `pipeline.py` always instantiates params explicitly (never relying on defaults alone).
4. `self.step_params` is always a typed `XxxParams` instance (or `None` for param-free steps).
5. `task.connect()` receives `self.step_params.model_dump()` — ClearML UI override behaviour is unchanged.
6. Never mutate `self.step_params` after init — use instance variables for runtime-computed values.

## How It Works

```python
# src/preprocess/preprocess_pipeline_step.py
class PreprocessParams(BaseModel):
    skip_mark: bool = False          # typed, IDE-complete, validated

class PreprocessPipelineStep(BasePipelineStep):
    def __init__(self, params: PreprocessParams):
        super().__init__(PREPROCESS, params=params)

    def start(self, dataset_id: str) -> str:
        if self.step_params.skip_mark:   # attribute access, not dict .get()
            ...
```

```python
# src/pipelines/pipeline.py
def run_preprocess_step(dataset_id: str) -> str:
    return PreprocessPipelineStep(
        params=PreprocessParams(skip_mark=False),
    ).start(dataset_id=dataset_id)
```

## Steps without parameters

Steps that need no parameters (e.g. `FeatureEngineerPipelineStep`) simply omit `params`:

```python
class FeatureEngineerPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(FEATURE_ENGINEER)   # params defaults to None
```

`_connect_params()` is a no-op when `self.step_params is None`.

## Agent Checklist

Before adding or modifying parameters:
- [ ] Is `XxxParams(BaseModel)` defined in the same file as the step?
- [ ] Is `params` a required constructor argument (no default)?
- [ ] Does `pipeline.py` pass params explicitly at the call site?
- [ ] Does step code use `self.step_params.field_name` (not `.get()`)?
- [ ] Are runtime-computed values stored as instance vars (not mutating `self.step_params`)?
```

- [ ] **Step 2: Commit**

```bash
git add docs/knowledge/generated-project/params-yaml-contract.md
git commit -m "docs: update params contract to reflect explicit Pydantic model injection"
```
