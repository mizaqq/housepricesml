import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, GridSearchCV


def perform_grid_search(
    output_path: Path = Path(__file__).parent.parent.resolve().joinpath("data", "xgb-grid-search-results-01.csv"),
    test_config_path: Path = Path(__file__).parent.parent.resolve().joinpath("data", "grid_search_params.json"),
    test_size: float = 0.2,
    random_state: int = 0,
    target_variable: str = "SalePrice",
    data_path: Path = Path(__file__).parent.parent.resolve().joinpath("data", "train_preprocessed.csv"),
) -> None:
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(target_variable, axis=1))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_variable, axis=1), df[target_variable], test_size=test_size, random_state=random_state
    )
    config = json.load(open(test_config_path, "r"))
    params = config["params"]
    # TODO input model from config
    model_test = XGBRegressor(random_state=random_state)
    skf = KFold(n_splits=config["splits"], shuffle=config["shuffle"], random_state=random_state)
    grid = GridSearchCV(model_test, param_grid=params, scoring=config["scoring"], n_jobs=-1, cv=skf, verbose=4)
    grid.fit(X_train, y_train)

    print("\n All results:")
    print(grid.cv_results_)
    print("\n Best estimator:")
    print(grid.best_estimator_)
    print("\n Best score:")
    print(grid.best_score_ * 2 - 1)
    print("\n Best parameters:")
    print(grid.best_params_)
    results = pd.DataFrame(grid.cv_results_)
    results.to_csv(output_path, index=False)

    y_test = grid.best_estimator_.predict_proba(y_test)
    results_df = pd.DataFrame(data={"id": y_test["id"], "target": y_test[:, 1]})
    results_df.to_csv("score_on_test" + output_path, index=False)


def perform_random_search(
    X_train,
    y_train,
    output_path: Path = Path(__file__).parent.parent.resolve().joinpath("data", "best_params_random_search.json"),
    rendom_state: int = 0,
    test_config_path: Path = Path(__file__).parent.parent.resolve().joinpath("data", "random_search_params.json"),
) -> dict:
    config = json.load(open(test_config_path, "r"))
    params = config["params"]

    model_test = XGBRegressor(random_state=rendom_state)

    skf = KFold(n_splits=config["splits"], shuffle=True, random_state=rendom_state)
    random_search = RandomizedSearchCV(
        model_test,
        param_distributions=params,
        n_iter=config["n_iter"],
        scoring=config["scoring"],
        n_jobs=-1,
        cv=skf,
        verbose=3,
        random_state=rendom_state,
    )
    random_search.fit(X_train, y_train)
    with open(output_path, "w") as f:
        json.dump(random_search.best_params_, f)
    return random_search.best_params_
