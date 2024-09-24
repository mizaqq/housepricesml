from model.utils.data_preprocess import preprocess_data, split_data, normalize_data, scale_data
from model.models import models
from keras.api.layers import Dense
import pandas as pd
import argparse
from model.utils.hyper_params import perform_random_search
import logging
import json

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="regressor")


def main(args):
    df = pd.read_csv("model/data/train.csv")
    df = preprocess_data(df)
    # df = normalize_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    if args.model == "regressor":
        model = models.Regressor()
    elif args.model == "xgb":
        with open('best_params.json', 'r') as f:
            params = json.load(f)
        model = models.XGBModel(**params)
    elif args.model == "neural_network":
        model = models.NeuralNetwork()
        model.add_to_model(Dense(64, activation="relu"))
        model.add_to_model(Dense(32, activation="relu"))
        model.add_to_model(Dense(1, activation="linear"))
        model.compile()
    model.fit(X_train, y_train)
    print(model.evaluate(X_test, y_test))


args = argparser.parse_args()
args.model = "xgb"
main(args)
