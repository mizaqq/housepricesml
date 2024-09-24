from pydantic import BaseModel
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from keras.api.models import Sequential
from keras.api import Layer
from sklearn.metrics import mean_squared_error


class Model(BaseModel, ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def evaluate(self, predictions: pd.DataFrame, y: pd.DataFrame):
        pass


class Regressor(Model):
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame):
        return mean_squared_error(y, self.predict(X))


class XGBModel(Model):
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame):
        return mean_squared_error(y, self.predict(X))


class NeuralNetwork(Model):
    def __init__(self, **kwargs):
        self.model = Sequential()

    def add_to_model(self, layer: Layer):
        self.model.add(layer)

    def compile(self, optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame):
        return mean_squared_error(y, self.predict(X))
