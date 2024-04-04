# -*- coding: utf-8 -*-
"""Module defines enums."""
from common.pipeline_steps import (
    PRE_RUN,
    PREPROCESS,
    FEATURE_ENGINEER,
    SELECT_FEATURES,
    SPLIT_DATASET,
    TRAIN,
    PLOTTING,
    HYPERPARAMETER_OPTIMIZATION,
    POST_RUN,
)


class PipelineSteps:
    """
    Enum for pipeline steps.
    """
    pre_run = PRE_RUN
    preprocess = PREPROCESS
    feature_engineer = FEATURE_ENGINEER
    select_features = SELECT_FEATURES
    split_dataset = SPLIT_DATASET
    train = TRAIN
    plotting = PLOTTING
    hyperparameter_optimization = HYPERPARAMETER_OPTIMIZATION
    post_run = POST_RUN