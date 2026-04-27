# -*- coding: utf-8 -*-
"""FEATURE_ENGINEER pipeline step — coordinator that fans out to per-file worker tasks."""
from pydantic import BaseModel

from src.common.pipeline_steps import FEATURE_ENGINEER
from src.core.coordinator_step import BaseCoordinatorStep
from src.settings import SETTINGS


class FeatureEngineerParams(BaseModel):
    pass


class FeatureEngineerPipelineStep(BaseCoordinatorStep):
    def __init__(self, params: FeatureEngineerParams = None) -> None:
        super().__init__(FEATURE_ENGINEER, params=params or FeatureEngineerParams())

    def worker_entry_point(self) -> str:
        return "src/features/worker.py"

    def _queue_name(self) -> str:
        return SETTINGS.clearml.feature_engineer_queue
