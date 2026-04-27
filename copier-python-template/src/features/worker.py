# -*- coding: utf-8 -*-
"""Standalone ClearML worker task for feature engineering a single file."""
import tempfile
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import yaml
from sklearn import set_config

from src.core.worker_step import BaseWorkerStep
from src.settings import SETTINGS

set_config(transform_output="pandas")


class FeatureEngineerWorkerStep(BaseWorkerStep):
    def process(self, input_path: str, output_path: str) -> None:
        bucket = SETTINGS.object_storage.bucket
        s3 = boto3.client(
            "s3",
            endpoint_url=SETTINGS.object_storage.endpoint,
            aws_access_key_id=SETTINGS.object_storage.access_key.get_secret_value(),
            aws_secret_access_key=SETTINGS.object_storage.secret_key.get_secret_value(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download input parquet
            local_input = Path(tmpdir) / Path(input_path).name
            in_key = input_path.removeprefix(f"s3://{bucket}/")
            s3.download_file(bucket, in_key, str(local_input))

            # Download labeling config
            config_key = (
                f"{SETTINGS.storage.labels_folder.name}"
                f"/{SETTINGS.labeling_config_path.name}"
            )
            local_config = Path(tmpdir) / SETTINGS.labeling_config_path.name
            s3.download_file(bucket, config_key, str(local_config))

            # Load config and apply feature engineering
            with open(local_config) as f:
                labeling_config = yaml.safe_load(f)

            data = pd.read_parquet(local_input)
            fe = _build_feature_engineer(labeling_config)
            result: pd.DataFrame = fe.fit_transform(data)

            # Upload result
            local_output = Path(tmpdir) / Path(output_path).name
            result.to_parquet(local_output, index=False)
            out_key = output_path.removeprefix(f"s3://{bucket}/")
            s3.upload_file(str(local_output), bucket, out_key)


def _build_feature_engineer(labeling_config: dict[str, Any]) -> "FeatureEngineer":  # type: ignore[name-defined]
    # labeling_config reserved for future use; FeatureEngineer reads its
    # config from SETTINGS at construction time.
    from src.features.feature_engineer import FeatureEngineer
    return FeatureEngineer()


if __name__ == "__main__":
    FeatureEngineerWorkerStep.main()
