# -*- coding: utf-8 -*-
"""Module describes main features in the project."""
from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class Feature:
    """Dataclass for describing features"""

    name: str
    dtype: str
    description: Optional[str] = None
    lower: Optional[Union[int, float]] = None
    upper: Optional[Union[int, float]] = None
    fillna_value: Optional[Union[int, float]] = None
    fillna_method: Optional[str] = "ffill"
    fillna_limit: Optional[int] = None

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __contains__(self, item: Union[str, "Feature", None]):
        if item is None:
            return False
        else:
            return str(item) in self.name


FEATURE_1 = Feature(
    name="feature_1",
    dtype="float32",
    lower=0,
    upper=1000,
    fillna_value=0,
)
FEATURE_2 = Feature(
    name="feature_2",
    dtype="float64",
    lower=0,
    upper=25,
    fillna_value=0,
)
FEATURE_3 = Feature(
    name="feature_3",
    dtype="float32",
    lower=0,
    upper=1000,
    fillna_value=0,
)


IGNORED_FEATURES = [FEATURE_1]
MANDATORY_FEATURES = [
    FEATURE_1,
    FEATURE_2,
    FEATURE_3,
]
