from model.utils.data_preprocess import preprocess_data
from model.models import models
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="regressor")


def main(args):
    df = pd.read_csv("data/train.csv")
    df = preprocess_data(df)
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    if args.model == "regressor":
        model = models.Regressor().fit(X, y)
    elif args.model == "xgb":
        model = models.XGBModel().fit(X, y)
    elif args.model == "neural_network":
        model = models.NeuralNetwork()
        model.add_to_model(models.Dense(64, activation="relu"))
        model.add_to_model(models.Dense(32, activation="relu"))
        model.add_to_model(models.Dense(1, activation="linear"))
        model.compile()
        model.fit(X, y)
    print(model.evaluate(X, y))


main(args=argparser.parse_args())
