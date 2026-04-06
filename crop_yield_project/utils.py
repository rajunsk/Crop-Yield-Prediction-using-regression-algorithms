# =============================================================================
# utils.py — Crop Yield Prediction Project
# Shared utility functions used across all notebooks/scripts
# =============================================================================

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error, mean_absolute_percentage_error
)
from config import FIGURE_DPI, FIGURE_STYLE, OUTPUT_DIR, REPORTS_DIR

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── I/O Helpers ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load CSV with logging."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols from {os.path.basename(path)}")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV with logging."""
    df.to_csv(path, index=False)
    logger.info(f"Saved {df.shape[0]:,} rows → {os.path.basename(path)}")


def save_model(obj, path: str) -> None:
    """Pickle any Python object."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved model → {os.path.basename(path)}")


def load_model(path: str):
    """Load pickled object."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded model ← {os.path.basename(path)}")
    return obj


def ensure_dirs() -> None:
    """Create output / report directories if they don't exist."""
    for d in [OUTPUT_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, label: str = "Model") -> dict:
    """
    Compute and print regression metrics.
    Works on log-scale predictions — converts back to original scale internally.
    """
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    metrics = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    print(f"\n{'='*45}")
    print(f"  {label} — Evaluation Metrics")
    print(f"{'='*45}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"{'='*45}\n")
    return metrics


def compare_models(results: dict) -> pd.DataFrame:
    """
    Display a comparison table for multiple models.
    results = {model_name: metrics_dict}
    """
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["Model", "R²", "MAE", "RMSE", "MAPE (%)"]
    df = df.sort_values("R²", ascending=False).reset_index(drop=True)
    print("\n" + df.to_string(index=False))
    return df


# ─── Cleaning Helpers ────────────────────────────────────────────────────────
def cap_outliers_iqr(df: pd.DataFrame, col: str, multiplier: float = 3.0) -> pd.DataFrame:
    """Cap extreme values using IQR × multiplier."""
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
    before = df[col].describe()[["min", "max"]].to_dict()
    df[col] = df[col].clip(lower=lower, upper=upper)
    after  = df[col].describe()[["min", "max"]].to_dict()
    logger.info(f"  {col}: capped [{before['min']:.2f}, {before['max']:.2f}] → [{after['min']:.2f}, {after['max']:.2f}]")
    return df


def log1p_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Apply log1p to selected columns (handles zeros safely)."""
    for col in cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
            logger.info(f"  log1p applied → log_{col}")
    return df


# ─── Feature Engineering Helpers ─────────────────────────────────────────────
def add_rainfall_zone(df: pd.DataFrame,
                      bins: list, labels: list,
                      col: str = "Annual_Rainfall") -> pd.DataFrame:
    """Bin rainfall into categorical zones."""
    df["Rainfall_Zone"] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df


def add_efficiency_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable ratio features."""
    df["Fertilizer_per_Area"] = df["Fertilizer"] / (df["Area"] + 1e-6)
    df["Pesticide_per_Area"]  = df["Pesticide"]  / (df["Area"] + 1e-6)
    df["Production_per_Area"] = df["Production"] / (df["Area"] + 1e-6)
    df["Rainfall_per_Area"]   = df["Annual_Rainfall"] / (df["Area"] + 1e-6)
    df["Fertilizer_to_Pesticide"] = df["Fertilizer"] / (df["Pesticide"] + 1e-6)
    logger.info("  Added 5 efficiency ratio features")
    return df


def add_temporal_features(df: pd.DataFrame, year_col: str = "Crop_Year") -> pd.DataFrame:
    """Add decade and era features."""
    df["Decade"] = (df[year_col] // 10) * 10
    df["Era"] = pd.cut(
        df[year_col],
        bins=[1996, 2004, 2012, 2021],
        labels=["1997-2004", "2005-2012", "2013-2020"]
    )
    logger.info("  Added Decade and Era temporal features")
    return df


# ─── Plotting Helpers ────────────────────────────────────────────────────────
def set_style() -> None:
    try:
        plt.style.use(FIGURE_STYLE)
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi"    : FIGURE_DPI,
        "font.size"     : 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def save_fig(fig, name: str, subdir: str = "eda") -> str:
    """Save a matplotlib figure to the reports folder."""
    folder = os.path.join(REPORTS_DIR, subdir)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"  Plot saved → {os.path.relpath(path)}")
    return path


def plot_actual_vs_predicted(y_true, y_pred, title: str = "Actual vs Predicted") -> None:
    """Scatter plot of true vs predicted values."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=15, color="#2d6a4f", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Fit")
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel("Actual Yield (log scale)")
    ax.set_ylabel("Predicted Yield (log scale)")
    ax.set_title(f"{title}\nR² = {r2:.4f}")
    ax.legend()
    save_fig(fig, "actual_vs_predicted", subdir="model")


def plot_residuals(y_true, y_pred) -> None:
    """Residual plot."""
    set_style()
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=12, color="#1b4332")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=50, color="#40916c", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    save_fig(fig, "residual_plots", subdir="model")


def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> None:
    """Bar chart of feature importances."""
    set_style()
    imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.sort_values().plot.barh(ax=ax, color="#52b788")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    save_fig(fig, "feature_importance", subdir="model")
