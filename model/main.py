import os
import logging
import pandas as pd
from keras.api.layers import Dense
from model.utils.data_preprocess import preprocess_data, split_data, normalize_data, get_data_for_preprocessing
from model.models.models import XGBModel, Regressor, NeuralNetwork, ModelType
from model.utils.mlflow import MLFlowHandler
from omegaconf import DictConfig, OmegaConf
import hydra

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://48.209.80.111:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "housepricesml")
    mlflow_handler = MLFlowHandler(uri, experiment_name)
    df = pd.read_csv(cfg["data"]["train"])
    uncorrelated_col, insignificant_col, missing_values_col = get_data_for_preprocessing(
        df, treshhold=cfg["preprocessing"]["threshhold"], mlflow=mlflow_handler
    )
    df = preprocess_data(df, uncorrelated_col, insignificant_col, missing_values_col)
    mlflow_handler.log_preprocessed_data(df)
    # df = normalize_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model_type = ModelType(cfg["model"]["type"].lower())
    if model_type.value == ModelType.REGRESSOR.value:
        model = Regressor()
    elif model_type.value == ModelType.XGB.value:
        params = cfg["model"]["params"]
        model = XGBModel(**params)
    else:
        model = NeuralNetwork()
        model.add_to_model(Dense(64, activation="relu"))
        model.add_to_model(Dense(32, activation="relu"))
        model.add_to_model(Dense(1, activation="linear"))
        model.compile()
    model.fit(X_train, y_train)
    score = model.evaluate(X_test, y_test)
    mlflow_handler.log_model(model, ("r2_score", score), X_train, model.get_params(), cfg["data"]["train"])
    mlflow_handler.close()


if __name__ == "__main__":
    main()
