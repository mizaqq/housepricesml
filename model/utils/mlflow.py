import os
import mlflow
import pandas as pd
import tempfile
from typing import Tuple, Sequence, Any, Optional, List
from pathlib import Path
from model.models.models import Model, Regressor, XGBModel
import subprocess


class MLFlowHandler:
    def __init__(self, uri: str, experiment_name: str) -> None:
        self.login_to_mlflow(uri, experiment_name)
        self.run = mlflow.start_run()

    def close(self) -> None:
        mlflow.end_run()

    def login_to_mlflow(self, uri: str, experiment_name: str) -> None:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

    def log_preprocessed_data(self, df: pd.DataFrame) -> None:
        with tempfile.NamedTemporaryFile() as temp:
            df.to_csv(temp.name)
            mlflow.log_artifact(temp.name)

    def log_analysis(self, data: Sequence[Tuple[Any, ...]], metric_type: str) -> None:
        for item in data:
            mlflow.log_metric(item[0] + " " + metric_type, item[1])

    def log_commit(self) -> None:
        tag = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
        mlflow.set_tag("Git Commit", tag.decode())

    def log_model(
        self,
        model: Model,
        score: Tuple[str, float],
        X_train: Sequence,
        params: Optional[dict] = None,
        artifact: Optional[Path] = None,
    ) -> None:
        if params is not None:
            for key, value in params.items():
                mlflow.log_param(key, value)
        mlflow.log_metric(score[0], score[1])
        mlflow.set_tag("".join(ch for ch in str(type(model)).split(".")[-1] if ch.isalnum()), "training")
        if artifact is not None:
            mlflow.log_artifact(artifact)
        if type(model) is Regressor:
            mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path="houseprices_model",
                input_example=X_train[:10],
                registered_model_name="houseprices.default.Regressor",
            )

        elif type(model) is XGBModel:
            mlflow.xgboost.log_model(
                xgb_model=model.model,
                artifact_path="houseprices_model",
                input_example=X_train[:10],
                registered_model_name="houseprices.default.XGB",
            )
        else:
            mlflow.tensorflow.log_model(
                model=model.model, artifact_path="houseprices_model", input_example=X_train[:10]
            )
