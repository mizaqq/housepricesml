import logging
import json
import argparse
import os
import pandas as pd
from pathlib import Path
from keras.api.layers import Dense
from model.utils.data_preprocess import preprocess_data, split_data, normalize_data, scale_data
from model.models.models import XGBModel, Regressor, NeuralNetwork, ModelType
from model.utils.hyper_params import perform_random_search

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default=ModelType.NEURAL_NETWORK)
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


def main(args):
    df = preprocess_data(pd.read_csv(args.train_data))
    # df = normalize_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model_type = ModelType(args.model)
    if model_type.value == ModelType.REGRESSOR.value:
        model = Regressor()
        params = model.model.get_params()
    elif model_type.value == ModelType.XGB.value:
        with open(args.config, "r") as f:
            params = json.load(f)
        model = XGBModel(**params)
        params = model.model.get_params()
    else:
        model = NeuralNetwork()
        model.add_to_model(Dense(64, activation="relu"))
        model.add_to_model(Dense(32, activation="relu"))
        model.add_to_model(Dense(1, activation="linear"))
        model.compile()
        params = model.model.get_config()
    model.fit(X_train, y_train)
    score = model.evaluate(X_test, y_test)
    model.logmodel(model_type, ("r2_score", score), params=params, artifact=args.train_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
