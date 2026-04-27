# -*- coding: utf-8 -*-
"""PREPROCESS pipeline step — coordinator that fans out to per-file worker tasks."""
from pydantic import BaseModel

from src.common.pipeline_steps import PREPROCESS
from src.core.coordinator_step import BaseCoordinatorStep
from src.settings import SETTINGS


class PreprocessParams(BaseModel):
    skip_mark: bool = False


class PreprocessPipelineStep(BaseCoordinatorStep):
    def __init__(self, params: PreprocessParams) -> None:
        super().__init__(PREPROCESS, params=params)

    def worker_entry_point(self) -> str:
        return "src/preprocess/worker.py"

    def _queue_name(self) -> str:
        return SETTINGS.clearml.preprocess_queue
