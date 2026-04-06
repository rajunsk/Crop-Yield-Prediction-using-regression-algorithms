# =============================================================================
# config.py — Crop Yield Prediction Project
# Central configuration file for all scripts
# =============================================================================

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

RAW_DATA_PATH     = os.path.join(DATA_DIR, "crop_yield.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "crop_yield_cleaned.csv")
FEATURED_DATA_PATH = os.path.join(DATA_DIR, "crop_yield_featured.csv")
MODEL_PATH        = os.path.join(OUTPUT_DIR, "best_model.pkl")
ENCODER_PATH      = os.path.join(OUTPUT_DIR, "encoders.pkl")
SCALER_PATH       = os.path.join(OUTPUT_DIR, "scaler.pkl")

# ─── Dataset Columns ──────────────────────────────────────────────────────────
TARGET_COL   = "Yield"
CAT_COLS     = ["Crop", "Season", "State"]
NUM_COLS     = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]
YEAR_COL     = "Crop_Year"

# ─── Cleaning Parameters ─────────────────────────────────────────────────────
YIELD_LOWER_PCT   = 0.01   # 1st percentile — remove near-zero yields
YIELD_UPPER_PCT   = 0.99   # 99th percentile — remove extreme outliers
OUTLIER_IQR_MULT  = 3.0    # IQR multiplier for numeric feature outlier capping

# ─── Feature Engineering ─────────────────────────────────────────────────────
LOG_TRANSFORM_COLS = ["Area", "Production", "Fertilizer", "Pesticide", "Yield"]
SEASONS = ["Kharif", "Rabi", "Whole Year", "Autumn", "Summer", "Winter"]

# Rainfall zone thresholds (mm)
RAINFALL_BINS   = [0, 500, 1000, 1500, 2500, 9999]
RAINFALL_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]

# ─── Model Parameters ────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

XGBOOST_PARAMS = {
    "n_estimators"    : 500,
    "learning_rate"   : 0.05,
    "max_depth"       : 6,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
    "random_state"    : RANDOM_STATE,
    "n_jobs"          : -1,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth"   : 12,
    "min_samples_split": 5,
    "min_samples_leaf" : 2,
    "random_state": RANDOM_STATE,
    "n_jobs"      : -1,
}

# ─── EDA Plot Settings ───────────────────────────────────────────────────────
FIGURE_DPI    = 150
FIGURE_STYLE  = "seaborn-v0_8-whitegrid"
COLOR_PALETTE = "viridis"
TOP_N_CROPS   = 15
TOP_N_STATES  = 10

# ─── Power BI Export ─────────────────────────────────────────────────────────
POWERBI_EXPORT_PATH = os.path.join(OUTPUT_DIR, "crop_yield_powerbi.csv")
