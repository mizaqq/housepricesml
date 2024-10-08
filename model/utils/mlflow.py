import os
import logging
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from typing import Tuple, List
from pathlib import Path
from model.models.models import ModelType, Regressor, NeuralNetwork, XGBModel


class MLFlowHandler:
    def __init__(self):
        self.login_to_mlflow()
        self.run = mlflow.start_run()

    def close(self):
        mlflow.end_run()

    def login_to_mlflow(
        self, uri: str = "databricks", experiment_name: str = "/Users/michal.zareba@softwaremill.pl/housepricesml"
    ) -> None:
        mlflow.login()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        mlflow.set_registry_uri("databricks-uc")

    def log_preprocessed_data(self, df: pd.DataFrame) -> None:
        df.to_csv("preprocessed_data.csv")
        mlflow.log_artifact("preprocessed_data.csv")
        os.remove("preprocessed_data.csv")

    def log_analysis(self, data: List[Tuple], metric_type: str) -> None:
        for item in data:
            mlflow.log_metric(item[0] + " " + metric_type, item[1])

    def log_model(
        self,
        model: ModelType,
        score: Tuple[str, float],
        params: dict = None,
        artifact: Path = None,
        X_train: pd.DataFrame = None,
    ) -> None:
        if params is not None:
            for key, value in params.items():
                mlflow.log_param(key, value)
        mlflow.log_metric(score[0], score[1])
        mlflow.set_tag("".join(ch for ch in str(type(model)).split(".")[-1] if ch.isalnum()), "training")
        if artifact is not None:
            mlflow.log_artifact(artifact)
        if type(model) == Regressor:
            mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path="houseprices_model",
                input_example=X_train[:10],
                registered_model_name="houseprices.default.Regressor",
            )

        elif type(model) == XGBModel:
            mlflow.xgboost.log_model(
                xgb_model=model.model,
                artifact_path="houseprices_model",
                input_example=X_train[:10],
                registered_model_name="houseprices.default.XGB",
            )
        else:
            mlflow.keras.log_model(
                keras_model=model.model,
                artifact_path="houseprices_model",
                input_example=X_train[:10],
                registered_model_name="houseprices.default.NN",
            )
