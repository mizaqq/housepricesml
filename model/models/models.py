from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from keras.api.models import Sequential
from keras.api import Layer
from sklearn.metrics import mean_squared_error, r2_score
from typing import Union


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Model(ABC, CustomBaseModel):
    model: Union[LinearRegression, XGBRegressor, Sequential]

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
        super().__init__(model=LinearRegression())

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame):
        return r2_score(y, self.predict(X))


class XGBModel(Model):
    def __init__(self, **kwargs):
        super().__init__(model=XGBRegressor())
        self.model.set_params(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame):
        return r2_score(y, self.predict(X))


class NeuralNetwork(Model):
    def __init__(self):
        super().__init__(model=Sequential())

    def add_to_model(self, layer: Layer):
        self.model.add(layer)

    def compile(self, optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# TODO refactor this to intake fit params
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y, epochs=100, batch_size=32)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame):
        return r2_score(y, self.predict(X))


model = XGBModel()
