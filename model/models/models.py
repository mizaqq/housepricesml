import pandas as pd
from typing import Union, Tuple
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
from enum import Enum
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from keras.api.models import Sequential
from keras.api import Layer


class ModelType(Enum):
    REGRESSOR: str = "regressor"
    XGB: str = "xgb"
    NEURAL_NETWORK: str = "neural_network"


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Model(ABC, CustomBaseModel):
    model: Union[LinearRegression, XGBRegressor, Sequential]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def evaluate(self, predictions: pd.DataFrame, y: pd.DataFrame):
        pass


class Regressor(Model):
    def __init__(self, **kwargs) -> Model:
        super().__init__(model=LinearRegression())

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame) -> float:
        return r2_score(y, self.predict(X))


class XGBModel(Model):
    def __init__(self, **kwargs) -> Model:
        super().__init__(model=XGBRegressor())
        self.model.set_params(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame) -> float:
        return r2_score(y, self.predict(X))


class NeuralNetwork(Model):
    def __init__(self) -> Model:
        super().__init__(model=Sequential())

    def add_to_model(self, layer: Layer):
        self.model.add(layer)

    def compile(self, optimizer="adam", loss="mean_squared_error", metrics=["r2_score"]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> None:
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def evaluate(self, X, y: pd.DataFrame) -> float:
        return r2_score(y, self.predict(X))
