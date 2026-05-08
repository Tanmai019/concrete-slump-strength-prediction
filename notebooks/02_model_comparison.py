"""
Model comparison workflow for concrete slump prediction.

This script is a local, GitHub-friendly version of the original Colab notebook.
It removes notebook-only commands such as `!pip install lightgbm` and uses
relative paths.

Run from the project root:
    python src/02_model_comparison.py

Outputs:
    outputs/model_comparison_metrics.csv
    outputs/feature_importance_lightgbm.csv
    outputs/feature_importance_xgboost.csv
    outputs/plots/*.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cement_slump.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and standardize column names for the concrete slump dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place cement_slump.csv inside the data/ folder."
        )

    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "SLUMP(cm)": "SLUMP (cm)",
            "FLOW(cm)": "FLOW (cm)",
            "Compressive Strength (28-day)(Mpa)": "Compressive Strength (MPa)",
        }
    )

    expected_cols = [
        "Cement",
        "Slag",
        "Fly ash",
        "Water",
        "SP",
        "Coarse Aggr.",
        "Fine Aggr.",
        "SLUMP (cm)",
        "FLOW (cm)",
        "Compressive Strength (MPa)",
    ]

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[expected_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def evaluate_model(name: str, y_true, y_pred) -> dict:
    """Return regression metrics for a model."""
    return {
        "model": name,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train multiple regression models and return metrics."""
    X = df.drop("SLUMP (cm)", axis=1)
    y = df["SLUMP (cm)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }

    metrics = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics.append(evaluate_model(name, y_test, preds))
        fitted_models[name] = model

    lgb_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [2, 3, 4],
        "num_leaves": [10, 15, 20],
        "learning_rate": [0.05, 0.1],
    }

    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    lgb_grid = GridSearchCV(
        lgb_model,
        lgb_param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=0,
        n_jobs=-1,
    )
    lgb_grid.fit(X_train, y_train)
    best_lgb = lgb_grid.best_estimator_
    lgb_preds = best_lgb.predict(X_test)
    metrics.append(evaluate_model("LightGBM", y_test, lgb_preds))
    fitted_models["LightGBM"] = best_lgb

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        objective="reg:squarederror",
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    metrics.append(evaluate_model("XGBoost", y_test, xgb_preds))
    fitted_models["XGBoost"] = xgb_model

    ensemble = VotingRegressor(
        estimators=[
            ("random_forest", fitted_models["Random Forest"]),
            ("xgboost", xgb_model),
        ]
    )
    ensemble.fit(X_train, y_train)
    ensemble_preds = ensemble.predict(X_test)
    metrics.append(evaluate_model("Voting Ensemble", y_test, ensemble_preds))
    fitted_models["Voting Ensemble"] = ensemble

    metrics_df = pd.DataFrame(metrics).sort_values(by="rmse")
    return metrics_df, {
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "models": fitted_models,
    }


def save_outputs(metrics_df: pd.DataFrame, artifacts: dict) -> None:
    """Save model metrics and feature importance visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=metrics_df, x="rmse", y="model")
    plt.title("Model Comparison by RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_comparison_rmse.png", dpi=300)
    plt.close()

    X = artifacts["X"]
    best_lgb = artifacts["models"]["LightGBM"]
    xgb_model = artifacts["models"]["XGBoost"]

    lgb_importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": best_lgb.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    xgb_importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": xgb_model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    lgb_importance.to_csv(OUTPUT_DIR / "feature_importance_lightgbm.csv", index=False)
    xgb_importance.to_csv(OUTPUT_DIR / "feature_importance_xgboost.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=lgb_importance, x="Importance", y="Feature")
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance_lightgbm.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=xgb_importance, x="Importance", y="Feature")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance_xgboost.png", dpi=300)
    plt.close()


def main() -> None:
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    metrics_df, artifacts = train_models(df)
    save_outputs(metrics_df, artifacts)

    print("\nModel Comparison")
    print(metrics_df.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
