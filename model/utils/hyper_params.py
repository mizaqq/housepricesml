import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, GridSearchCV

def perform_grid_search():
    df = pd.read_csv("./model/data/train_preprocessed.csv")
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("SalePrice", axis=1))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("SalePrice", axis=1), df["SalePrice"], test_size=0.2, random_state=0
    )

    params = {
        "min_child_weight": [1, 5, 10],
        "gamma": [1, 1.5, 2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5, 6],
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "lambda": [0.01, 0.1],
        "alpha": [0.01, 0.1],
    }


    model_test = XGBRegressor(random_state=0)

    skf = KFold(n_splits=5, shuffle=True, random_state=0)
    grid = GridSearchCV(model_test, param_grid=params, scoring="r2", n_jobs=-1, cv=skf, verbose=4)
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
    results.to_csv("xgb-grid-search-results-01.csv", index=False)

    y_test = grid.best_estimator_.predict_proba(y_test)
    results_df = pd.DataFrame(data={"id": y_test["id"], "target": y_test[:, 1]})
    results_df.to_csv("submission-grid-search-xgb-porto-01.csv", index=False)



def perform_random_search(X_train,y_train):
  params = {
          'learning_rate': [0.01, 0.05, 0.1],
          'n_estimators': [100, 500, 1000],
          'min_child_weight': [1, 5, 10],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'subsample': [0.6, 0.8, 1.0],
          'colsample_bytree': [0.6, 0.8, 1.0],
          'max_depth': [3, 4, 5]
          }

  model_test = XGBRegressor(random_state=0)

  skf = KFold(n_splits=5, shuffle = True, random_state = 0)
  param_comb = 7
  random_search = RandomizedSearchCV(model_test, param_distributions=params, n_iter=param_comb, scoring='r2', n_jobs=-1, cv=skf, verbose=3, random_state=0 )
  random_search.fit(X_train, y_train)
  return random_search.best_params_