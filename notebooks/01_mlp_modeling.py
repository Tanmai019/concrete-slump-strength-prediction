"""
MLP modeling workflow for concrete slump prediction.

This script is a local, GitHub-friendly version of the original Colab notebook.
It expects the dataset at: data/cement_slump.csv

Run from the project root:
    python src/01_mlp_modeling.py

Outputs:
    outputs/mlp_metrics.csv
    outputs/mlp_feature_importance.csv
    outputs/plots/*.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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

    rename_map = {
        "SLUMP(cm)": "SLUMP (cm)",
        "FLOW(cm)": "FLOW (cm)",
        "Compressive Strength (28-day)(Mpa)": "Compressive Strength (MPa)",
    }
    df = df.rename(columns=rename_map)

    numeric_cols = [
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

    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def save_eda_plots(df: pd.DataFrame) -> None:
    """Save basic EDA plots to outputs/plots."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    for col in df.columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"distribution_{clean_filename(col)}.png", dpi=300)
        plt.close()

    for col in df.columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"boxplot_{clean_filename(col)}.png", dpi=300)
        plt.close()

    target = "SLUMP (cm)"
    for feature in df.columns.drop(target):
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df[feature], y=df[target])
        plt.title(f"{feature} vs {target}")
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"scatter_{clean_filename(feature)}_vs_slump.png", dpi=300)
        plt.close()


def clean_filename(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
    )


def train_mlp(df: pd.DataFrame):
    """Train and tune an MLPRegressor for slump prediction."""
    X = df.drop("SLUMP (cm)", axis=1)
    y = df["SLUMP (cm)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    }

    model = MLPRegressor(
        max_iter=2000,
        early_stopping=True,
        random_state=42,
    )

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    metrics = {
        "best_params": str(grid.best_params_),
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    return best_model, scaler, X, X_test_scaled, y_test, y_pred, metrics


def save_model_outputs(model, X, X_test_scaled, y_test, y_pred, metrics) -> None:
    """Save metrics, feature importance, and prediction plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "mlp_metrics.csv", index=False)

    result = permutation_importance(
        model,
        X_test_scaled,
        y_test,
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": result.importances_mean,
        }
    ).sort_values(by="Importance", ascending=False)

    importance_df.to_csv(OUTPUT_DIR / "mlp_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("MLP Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mlp_feature_importance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, edgecolors="k", alpha=0.75)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual Slump (cm)")
    plt.ylabel("Predicted Slump (cm)")
    plt.title("MLP Regressor: Actual vs Predicted Slump")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mlp_actual_vs_predicted.png", dpi=300)
    plt.close()


def main() -> None:
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    save_eda_plots(df)
    model, scaler, X, X_test_scaled, y_test, y_pred, metrics = train_mlp(df)
    save_model_outputs(model, X, X_test_scaled, y_test, y_pred, metrics)

    print("\nMLP Results")
    print(f"Best Params: {metrics['best_params']}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f} cm")
    print(f"MAE: {metrics['mae']:.4f} cm")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
