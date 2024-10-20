import os
import mlflow
import pandas as pd
from typing import Tuple, List, Union, Optional
from pathlib import Path
from model.models.models import ModelType, Regressor, XGBModel


class MLFlowHandler:
    def __init__(self):
        self.login_to_mlflow()
        self.run = mlflow.start_run()

    def close(self):
        mlflow.end_run()

    def login_to_mlflow(
        self, uri: str = "http://48.209.80.111:5000", experiment_name: str = "housepricesml"
    ) -> None:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

    def log_preprocessed_data(self, df: pd.DataFrame, path: Path = Path("preprocessed_data.csv")) -> None:
        df.to_csv(path)
        mlflow.log_artifact(path)
        os.remove(path)

    def log_analysis(self, data: List[Tuple[str, Union[float, bool]]], metric_type: str) -> None:
        for item in data:
            mlflow.log_metric(item[0] + " " + metric_type, item[1])

    def log_model(
        self,
        model: ModelType,
        score: Tuple[str, float],
        params: Optional[dict] = None,
        artifact: Optional[Path] = None,
        X_train: Optional[pd.DataFrame] = None,
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
