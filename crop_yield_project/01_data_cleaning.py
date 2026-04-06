# =============================================================================
# 01_data_cleaning.py — Crop Yield Prediction Project
# Data Cleaning, Transformation & Feature Engineering
# =============================================================================
# Output : data/crop_yield_cleaned.csv
#          data/crop_yield_featured.csv
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import config as cfg
from utils import (
    load_data, save_data, logger, ensure_dirs, set_style, save_fig,
    cap_outliers_iqr, log1p_transform,
    add_rainfall_zone, add_efficiency_ratios, add_temporal_features,
)

# ─── 0. Setup ─────────────────────────────────────────────────────────────────
ensure_dirs()
set_style()

# ─── 1. Load Raw Data ─────────────────────────────────────────────────────────
df = load_data(cfg.RAW_DATA_PATH)
print(f"\n{'='*55}")
print(f"  STEP 1 | Raw Data Shape : {df.shape}")
print(f"{'='*55}")
print(df.head())

# ─── 2. Basic Inspection ──────────────────────────────────────────────────────
print("\n[2] Data Types & Null Counts")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# ─── 3. Strip Whitespace from String Columns ─────────────────────────────────
print("\n[3] Stripping whitespace from categorical columns ...")
for col in cfg.CAT_COLS + [cfg.YEAR_COL]:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()
logger.info("  Whitespace stripped from: Crop, Season, State")

# Validate seasons
print(f"  Seasons found : {sorted(df['Season'].unique())}")
print(f"  Total Crops   : {df['Crop'].nunique()}")
print(f"  Total States  : {df['State'].nunique()}")
print(f"  Year Range    : {df['Crop_Year'].min()} – {df['Crop_Year'].max()}")

# ─── 4. Handle Zeros & Impossible Values ─────────────────────────────────────
print("\n[4] Handling zero/negative values ...")
initial_rows = len(df)

# Zero yields are not meaningful for a prediction task
zero_yield = (df[cfg.TARGET_COL] <= 0).sum()
df = df[df[cfg.TARGET_COL] > 0].copy()
logger.info(f"  Removed {zero_yield} rows with Yield ≤ 0")

# Zero area is physically impossible
zero_area = (df["Area"] <= 0).sum()
df = df[df["Area"] > 0].copy()
logger.info(f"  Removed {zero_area} rows with Area ≤ 0")

# ─── 5. Outlier Treatment on Yield (Percentile Capping) ──────────────────────
print("\n[5] Outlier capping on Yield ...")
lower_thresh = df[cfg.TARGET_COL].quantile(cfg.YIELD_LOWER_PCT)
upper_thresh = df[cfg.TARGET_COL].quantile(cfg.YIELD_UPPER_PCT)
before_count = len(df)
df = df[(df[cfg.TARGET_COL] >= lower_thresh) & (df[cfg.TARGET_COL] <= upper_thresh)].copy()
removed = before_count - len(df)
logger.info(f"  Yield range kept: [{lower_thresh:.3f}, {upper_thresh:.3f}]")
logger.info(f"  Removed {removed} extreme yield outliers")

# IQR capping for numeric features
print("\n  IQR-based capping on numeric features ...")
for col in ["Area", "Production", "Fertilizer", "Pesticide"]:
    df = cap_outliers_iqr(df, col, multiplier=cfg.OUTLIER_IQR_MULT)

rows_after_cleaning = len(df)
print(f"\n  Rows after cleaning: {rows_after_cleaning:,}  "
      f"(removed {initial_rows - rows_after_cleaning:,} rows total = "
      f"{(initial_rows - rows_after_cleaning)/initial_rows*100:.1f}%)")

# ─── 6. Save Cleaned Dataset ─────────────────────────────────────────────────
save_data(df, cfg.CLEANED_DATA_PATH)

# ─── 7. Feature Engineering ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  STEP 7 | Feature Engineering")
print(f"{'='*55}")

# 7a. Rainfall zone binning
df = add_rainfall_zone(df, bins=cfg.RAINFALL_BINS, labels=cfg.RAINFALL_LABELS)
print(f"\nRainfall Zone Distribution:\n{df['Rainfall_Zone'].value_counts()}")

# 7b. Efficiency ratios
df = add_efficiency_ratios(df)

# 7c. Temporal features
df = add_temporal_features(df, year_col=cfg.YEAR_COL)

# 7d. Log1p transforms (handle skew)
log_cols = [c for c in cfg.LOG_TRANSFORM_COLS if c in df.columns]
df = log1p_transform(df, log_cols)

# 7e. Season order (ordinal — growing intensity proxy)
season_order = {
    "Kharif": 1, "Rabi": 2, "Whole Year": 3,
    "Autumn": 4, "Summer": 5, "Winter": 6
}
df["Season_Code"] = df["Season"].map(season_order).fillna(0).astype(int)
logger.info("  Season ordinal encoding added")

# 7f. Label encoding for categorical columns
encoders = {}
for col in cfg.CAT_COLS:
    le = LabelEncoder()
    df[f"{col}_Enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    logger.info(f"  Label encoded: {col} ({le.classes_.shape[0]} classes)")

# 7g. Interaction feature
df["Fertilizer_x_Rainfall"] = df["log_Fertilizer"] * df["Annual_Rainfall"] / 1000
df["Area_x_Rainfall"]       = df["log_Area"] * df["Annual_Rainfall"] / 1000
logger.info("  Interaction features added: Fertilizer_x_Rainfall, Area_x_Rainfall")

# 7h. Rolling state-crop mean yield (historical context)
df = df.sort_values(["State", "Crop", "Crop_Year"]).reset_index(drop=True)
df["StateCrop_AvgYield"] = (
    df.groupby(["State", "Crop"])[cfg.TARGET_COL]
      .transform(lambda x: x.expanding().mean().shift(1))
)
df["StateCrop_AvgYield"] = df["StateCrop_AvgYield"].fillna(df[cfg.TARGET_COL].median())
logger.info("  Historical rolling mean yield per State-Crop added")

# ─── 8. Summary of Engineered Features ───────────────────────────────────────
print(f"\n[8] Final Feature Count: {df.shape[1]} columns")
print("\nAll Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:02d}. {col}")

# ─── 9. Save Featured Dataset ─────────────────────────────────────────────────
save_data(df, cfg.FEATURED_DATA_PATH)

# ─── 10. Quick Visual — Yield Distribution Before vs After ───────────────────
print("\n[10] Saving cleaning validation plots ...")

raw_df = load_data(cfg.RAW_DATA_PATH)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(raw_df[cfg.TARGET_COL].clip(upper=raw_df[cfg.TARGET_COL].quantile(0.99)),
             bins=60, color="#e76f51", edgecolor="white")
axes[0].set_title("Yield Distribution — Raw")
axes[0].set_xlabel("Yield")
axes[0].set_ylabel("Count")

axes[1].hist(df[cfg.TARGET_COL], bins=60, color="#2d6a4f", edgecolor="white")
axes[1].set_title("Yield Distribution — After Cleaning")
axes[1].set_xlabel("Yield")
axes[1].set_ylabel("Count")

plt.suptitle("Yield Distribution: Raw vs Cleaned", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save_fig(fig, "yield_distribution_comparison", subdir="cleaning")

# Log-transformed yield distribution
fig2, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["log_Yield"], bins=60, color="#52b788", edgecolor="white")
ax.set_title("log(Yield+1) Distribution — Approx. Normal After Transform")
ax.set_xlabel("log(Yield + 1)")
ax.set_ylabel("Frequency")
plt.tight_layout()
save_fig(fig2, "log_yield_distribution", subdir="cleaning")

print(f"\n{'='*55}")
print(f"  DATA CLEANING COMPLETE")
print(f"  Cleaned rows   : {len(df):,}")
print(f"  Feature count  : {df.shape[1]}")
print(f"  Saved to       : {cfg.FEATURED_DATA_PATH}")
print(f"{'='*55}")
