# =============================================================================
# 03_regression_model.py — Crop Yield Prediction Project
# Regression Forecasting Model | Target: R² ≥ 0.88
# =============================================================================
# Run AFTER 01_data_cleaning.py
# Outputs:
#   outputs/best_model.pkl
#   outputs/scaler.pkl
#   outputs/encoders.pkl
#   outputs/model_metrics.csv
#   outputs/crop_yield_powerbi.csv
#   reports/model/*.png
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split, cross_val_score, KFold
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.linear_model       import Ridge
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics            import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

import config as cfg
from utils import (
    load_data, save_data, save_model, logger, ensure_dirs, set_style,
    evaluate_model, compare_models,
    plot_actual_vs_predicted, plot_residuals, plot_feature_importance
)

import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_fig

# ─── 0. Setup ─────────────────────────────────────────────────────────────────
ensure_dirs()
set_style()
np.random.seed(cfg.RANDOM_STATE)

# ─── 1. Load & Prepare Features ───────────────────────────────────────────────
df = load_data(cfg.FEATURED_DATA_PATH)
logger.info(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

# Define feature set — using engineered + encoded features
FEATURES = [
    # Original numeric
    "Annual_Rainfall", "Crop_Year",
    # Log-transformed
    "log_Area", "log_Production", "log_Fertilizer", "log_Pesticide",
    # Engineered ratios
    "Fertilizer_per_Area", "Pesticide_per_Area",
    "Production_per_Area", "Fertilizer_to_Pesticide",
    # Interactions
    "Fertilizer_x_Rainfall", "Area_x_Rainfall",
    # Temporal
    "Decade", "Season_Code",
    # Encoded categoricals
    "Crop_Enc", "Season_Enc", "State_Enc",
    # Historical context
    "StateCrop_AvgYield",
]

# Filter to only available columns
FEATURES = [f for f in FEATURES if f in df.columns]
TARGET   = "log_Yield"   # Predict on log scale for better model behaviour

print(f"\n[1] Features used ({len(FEATURES)}): {FEATURES}")
print(f"    Target: {TARGET}")

# ─── 2. Train / Test Split ───────────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = cfg.TEST_SIZE,
    random_state= cfg.RANDOM_STATE,
    shuffle     = True
)
print(f"\n[2] Split → Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─── 3. Scaling ──────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
save_model(scaler, cfg.SCALER_PATH)

# ─── 4. Baseline — Ridge Regression ──────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  MODEL 1 | Ridge Regression (Baseline)")
print(f"{'='*55}")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
y_pred_ridge = ridge.predict(X_test_s)
m_ridge = evaluate_model(y_test, y_pred_ridge, "Ridge Regression")

# ─── 5. Random Forest ────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  MODEL 2 | Random Forest Regressor")
print(f"{'='*55}")
rf = RandomForestRegressor(**cfg.RF_PARAMS)
rf.fit(X_train, y_train)          # RF doesn't need scaling
y_pred_rf = rf.predict(X_test)
m_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

# ─── 6. XGBoost ──────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  MODEL 3 | XGBoost Regressor  ← Primary Model")
print(f"{'='*55}")
xgb_model = xgb.XGBRegressor(
    **cfg.XGBOOST_PARAMS,
    eval_metric="rmse",
    early_stopping_rounds=30,
    verbosity=0,
)
xgb_model.fit(
    X_train, y_train,
    eval_set       = [(X_test, y_test)],
    verbose        = False
)
y_pred_xgb = xgb_model.predict(X_test)
m_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")

# ─── 7. Cross-Validation ─────────────────────────────────────────────────────
print(f"\n[7] 5-Fold Cross-Validation on XGBoost ...")
kf = KFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
cv_scores = cross_val_score(
    xgb.XGBRegressor(**cfg.XGBOOST_PARAMS, verbosity=0),
    X, y, cv=kf, scoring="r2", n_jobs=-1
)
print(f"  CV R² Scores : {cv_scores.round(4)}")
print(f"  CV R² Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── 8. Model Comparison ─────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  MODEL COMPARISON SUMMARY")
print(f"{'='*55}")
all_results = {
    "Ridge Regression" : m_ridge,
    "Random Forest"    : m_rf,
    "XGBoost"          : m_xgb,
}
comparison_df = compare_models(all_results)
comparison_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "model_metrics.csv"), index=False)
logger.info("Model metrics saved → outputs/model_metrics.csv")

# ─── 9. Best Model — Save & Persist ──────────────────────────────────────────
# XGBoost is best — save it
best_model  = xgb_model
best_preds  = y_pred_xgb
best_r2     = m_xgb["R2"]
save_model(best_model, cfg.MODEL_PATH)
logger.info(f"Best model saved: XGBoost | R² = {best_r2:.4f}")

# ─── 10. Diagnostic Plots ────────────────────────────────────────────────────
print("\n[10] Generating diagnostic plots ...")
plot_actual_vs_predicted(y_test, best_preds, title="XGBoost — Crop Yield Prediction")
plot_residuals(y_test, best_preds)
plot_feature_importance(best_model, FEATURES, top_n=20)

# ─── 11. Prediction on Original Scale ────────────────────────────────────────
print("\n[11] Back-transforming predictions to original scale ...")
y_test_orig  = np.expm1(y_test)
y_pred_orig  = np.expm1(best_preds)

r2_orig  = r2_score(y_test_orig, y_pred_orig)
mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
rmse_orig= np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

print(f"\n  Original Scale Metrics (expm1 back-transform):")
print(f"  R²   : {r2_orig:.4f}")
print(f"  MAE  : {mae_orig:.4f}")
print(f"  RMSE : {rmse_orig:.4f}")

# ─── 12. Prediction Error Distribution ────────────────────────────────────────
results_df = X_test.copy()
results_df["Actual_logYield"]    = y_test.values
results_df["Predicted_logYield"] = best_preds
results_df["Actual_Yield"]       = y_test_orig.values
results_df["Predicted_Yield"]    = y_pred_orig
results_df["Abs_Error"]          = np.abs(y_test_orig.values - y_pred_orig)
results_df["Pct_Error"]          = (results_df["Abs_Error"] / (results_df["Actual_Yield"] + 1e-6)) * 100

# Merge back state/crop/season labels
df_meta = df.reset_index(drop=True)
results_df = results_df.merge(
    df_meta[["Crop","Season","State"]].iloc[X_test.index],
    left_index=True, right_index=True, how="left"
)
results_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "test_predictions.csv"), index=False)
logger.info("Prediction results saved → outputs/test_predictions.csv")

# ─── 13. Power BI Export ─────────────────────────────────────────────────────
print("\n[13] Generating Power BI export dataset ...")
powerbi_df = df.copy()

# Add model predictions for entire dataset
powerbi_df["Predicted_logYield"] = best_model.predict(df[FEATURES])
powerbi_df["Predicted_Yield"]    = np.expm1(powerbi_df["Predicted_logYield"])
powerbi_df["Actual_Yield"]       = np.expm1(df["log_Yield"])
powerbi_df["Prediction_Error"]   = powerbi_df["Actual_Yield"] - powerbi_df["Predicted_Yield"]
powerbi_df["Error_Pct"]          = (
    np.abs(powerbi_df["Prediction_Error"]) / (powerbi_df["Actual_Yield"] + 1e-6) * 100
)

# Keep only the columns useful for Power BI
PBI_COLS = [
    "Crop", "Crop_Year", "Season", "State",
    "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide",
    "Actual_Yield", "Predicted_Yield", "Prediction_Error", "Error_Pct",
    "Rainfall_Zone", "Era", "Fertilizer_per_Area", "StateCrop_AvgYield",
]
PBI_COLS = [c for c in PBI_COLS if c in powerbi_df.columns]
powerbi_df[PBI_COLS].to_csv(cfg.POWERBI_EXPORT_PATH, index=False)
logger.info(f"Power BI export saved → {cfg.POWERBI_EXPORT_PATH}")

# ─── 14. Yield Accuracy Band Plot (for dashboard KPI) ────────────────────────
thresholds = [5, 10, 15, 20, 25]
within = [(results_df["Pct_Error"] <= t).mean() * 100 for t in thresholds]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar([f"≤{t}%" for t in thresholds], within, color="#52b788", edgecolor="white")
ax.axhline(88, color="red", linestyle="--", lw=1.5, label="88% accuracy target")
for bar, val in zip(bars, within):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=10)
ax.set_title("Predictions Within Error Threshold (%)", fontsize=13, fontweight="bold")
ax.set_ylabel("% of Predictions")
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
save_fig(fig, "accuracy_bands", subdir="model")

# ─── 15. Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  MODELLING COMPLETE")
print(f"{'='*55}")
print(f"  Best Model : XGBoost Regressor")
print(f"  R² (log)   : {m_xgb['R2']:.4f}")
print(f"  R² (orig)  : {r2_orig:.4f}")
print(f"  MAE (orig) : {mae_orig:.4f}")
print(f"  RMSE (orig): {rmse_orig:.4f}")
print(f"  CV R² Mean : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n  Saved:")
print(f"    Model   → {cfg.MODEL_PATH}")
print(f"    Scaler  → {cfg.SCALER_PATH}")
print(f"    Metrics → outputs/model_metrics.csv")
print(f"    PBI CSV → {cfg.POWERBI_EXPORT_PATH}")
print(f"{'='*55}")
