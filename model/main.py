import os
import logging
import json
import argparse
import pandas as pd
from pathlib import Path
from keras.api.layers import Dense
from model.utils.data_preprocess import preprocess_data, split_data, normalize_data, get_data_for_preprocessing
from model.models.models import XGBModel, Regressor, NeuralNetwork, ModelType
from model.utils.mlflow import MLFlowHandler

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default=ModelType.REGRESSOR)
    argparser.add_argument(
        "--train_data", default=Path(__file__).parent.resolve().joinpath("data", "train.csv"), type=str
    )
    argparser.add_argument(
        "--config",
        type=str,
        default=Path(__file__).parent.parent.resolve().joinpath("best_params.json"),
        help="Path to config file containing parameters for model",
    )
    return argparser.parse_args()


def main(args: argparse.Namespace) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://48.209.80.111:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "housepricesml")
    mlflow_handler = MLFlowHandler(uri, experiment_name)
    df = pd.read_csv(args.train_data)
    uncorrelated_col, insignificant_col, missing_values_col = get_data_for_preprocessing(
        df, treshhold=0.05, mlflow=mlflow_handler
    )
    df = preprocess_data(df, uncorrelated_col, insignificant_col, missing_values_col)
    mlflow_handler.log_preprocessed_data(df)
    # df = normalize_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model_type = ModelType(args.model)
    if model_type.value == ModelType.REGRESSOR.value:
        model = Regressor()
    elif model_type.value == ModelType.XGB.value:
        with open(args.config, "r") as f:
            params = json.load(f)
        model = XGBModel(**params)
    else:
        model = NeuralNetwork()
        model.add_to_model(Dense(64, activation="relu"))
        model.add_to_model(Dense(32, activation="relu"))
        model.add_to_model(Dense(1, activation="linear"))
        model.compile()
    model.fit(X_train, y_train)
    score = model.evaluate(X_test, y_test)
    mlflow_handler.log_model(model, ("r2_score", score), X_train, model.get_params(), args.train_data)
    mlflow_handler.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
