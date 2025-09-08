import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib
from pathlib import Path

def mean_r2(y_true, y_pred):
    return np.mean([r2_score(y_true.iloc[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

def mean_rmse(y_true, y_pred):
    return np.mean([np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i])) for i in range(y_true.shape[1])])

def train_models(X_csv: str, y_csv: str, out_dir_models: str, out_metrics_csv: str):
    X = pd.read_csv(X_csv)
    y = pd.read_csv(y_csv)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Linear Regression": MultiOutputRegressor(LinearRegression()),
        "SVR": MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=5000, early_stopping=True, random_state=42)
    }

    scoring = {
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "R2": make_scorer(mean_r2, greater_is_better=True),
        "RMSE": make_scorer(mean_rmse, greater_is_better=False)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_models = {}

    for name, model in models.items():
        scores = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring, return_estimator=True)
        results.append({
            "Model": name,
            "MAE (mean)": np.mean(scores["test_MAE"]),
            "R² (mean)": np.mean(scores["test_R2"]),
            "RMSE (mean)": np.mean(scores["test_RMSE"])
        })
        fitted_models[name] = scores["estimator"][0]

    results_df = pd.DataFrame(results).sort_values(by="R² (mean)", ascending=False)
    Path(out_dir_models).mkdir(parents=True, exist_ok=True)
    Path(out_metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_metrics_csv, index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, str(Path(out_dir_models) / "best_model.pkl"))
    joblib.dump(scaler, str(Path(out_dir_models) / "scaler.pkl"))
    return results_df, best_model_name
