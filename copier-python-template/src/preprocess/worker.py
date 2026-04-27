# -*- coding: utf-8 -*-
"""Standalone ClearML worker task for preprocessing a single file."""
from pathlib import Path

import boto3
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline

from src.core.worker_step import BaseWorkerStep
from src.settings import SETTINGS


class PreprocessWorkerStep(BaseWorkerStep):
    def process(self, input_path: str, output_path: str) -> None:
        params = self._get_worker_params()
        skip_mark: bool = bool(params.get("worker/skip_mark", False))
        bucket = SETTINGS.object_storage.bucket

        s3 = boto3.client(
            "s3",
            endpoint_url=SETTINGS.object_storage.endpoint,
            aws_access_key_id=SETTINGS.object_storage.access_key.get_secret_value(),
            aws_secret_access_key=SETTINGS.object_storage.secret_key.get_secret_value(),
        )

        local_input = Path("/tmp") / Path(input_path).name
        in_key = input_path.removeprefix(f"s3://{bucket}/")
        s3.download_file(bucket, in_key, str(local_input))

        data = pd.read_csv(local_input)
        steps: list = [("preprocessor", _build_preprocessor())]
        if not skip_mark:
            steps.append(("add_target", _build_mark_transformer()))
        set_config(transform_output="pandas")
        pipeline = Pipeline(steps=steps)
        result: pd.DataFrame = pipeline.transform(data)

        local_output = Path("/tmp") / Path(output_path).name
        result.to_parquet(local_output, index=False)
        out_key = output_path.removeprefix(f"s3://{bucket}/")
        s3.upload_file(str(local_output), bucket, out_key)


def _build_preprocessor():
    from src.preprocess.preprocessor import Preprocessor
    return Preprocessor()


def _build_mark_transformer():
    from src.preprocess.preprocessor import MarkDataTransformer
    return MarkDataTransformer()


if __name__ == "__main__":
    PreprocessWorkerStep.main()
