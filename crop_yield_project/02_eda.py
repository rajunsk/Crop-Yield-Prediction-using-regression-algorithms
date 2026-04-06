# =============================================================================
# 02_eda.py — Crop Yield Prediction Project
# Exploratory Data Analysis (EDA)
# =============================================================================
# Run AFTER 01_data_cleaning.py
# Outputs: reports/eda/*.png
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import config as cfg
from utils import load_data, logger, set_style, save_fig, ensure_dirs

# ─── 0. Setup ─────────────────────────────────────────────────────────────────
ensure_dirs()
set_style()

# Load cleaned featured dataset
df = load_data(cfg.FEATURED_DATA_PATH)

print(f"\n{'='*60}")
print(f"  EDA — Crop Yield Dataset  |  {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"{'='*60}")

# ─── EDA 1: Dataset Overview ──────────────────────────────────────────────────
print("\n[EDA-1] Basic Statistics")
print(df[[cfg.TARGET_COL, "Area", "Production", "Annual_Rainfall",
          "Fertilizer", "Pesticide"]].describe().round(2))

# ─── EDA 2: Yield Distribution ────────────────────────────────────────────────
print("\n[EDA-2] Yield Distribution")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df[cfg.TARGET_COL], bins=60, color="#52b788", edgecolor="white")
axes[0].set_title("Yield — Raw Scale")
axes[0].set_xlabel("Yield (tons/ha equivalent)")
axes[0].set_ylabel("Count")

axes[1].hist(df["log_Yield"], bins=60, color="#2d6a4f", edgecolor="white")
axes[1].set_title("log(Yield+1) — Near-Normal After Transform")
axes[1].set_xlabel("log(Yield + 1)")
axes[1].set_ylabel("Count")

plt.suptitle("Yield Distribution Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "01_yield_distribution", subdir="eda")

# ─── EDA 3: Yield by Season ───────────────────────────────────────────────────
print("\n[EDA-3] Yield by Season")
season_stats = df.groupby("Season")[cfg.TARGET_COL].agg(["mean","median","std"]).round(3)
print(season_stats)

fig, ax = plt.subplots(figsize=(10, 5))
season_order = df.groupby("Season")[cfg.TARGET_COL].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Season", y="log_Yield", order=season_order,
            palette="Set2", ax=ax, showfliers=False)
ax.set_title("log(Yield) Distribution by Season", fontsize=13, fontweight="bold")
ax.set_xlabel("Season")
ax.set_ylabel("log(Yield + 1)")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
save_fig(fig, "02_yield_by_season", subdir="eda")

# ─── EDA 4: Top Crops by Avg Yield ───────────────────────────────────────────
print("\n[EDA-4] Top Crops by Average Yield")
top_crops = (df.groupby("Crop")[cfg.TARGET_COL]
               .mean()
               .sort_values(ascending=False)
               .head(cfg.TOP_N_CROPS))
print(top_crops.round(3))

fig, ax = plt.subplots(figsize=(10, 6))
top_crops.sort_values().plot.barh(ax=ax, color="#74c69d", edgecolor="white")
ax.set_title(f"Top {cfg.TOP_N_CROPS} Crops by Average Yield", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean Yield")
plt.tight_layout()
save_fig(fig, "03_top_crops_by_yield", subdir="eda")

# ─── EDA 5: State-Wise Average Yield ─────────────────────────────────────────
print("\n[EDA-5] State-Wise Average Yield")
state_yield = (df.groupby("State")[cfg.TARGET_COL]
                 .mean()
                 .sort_values(ascending=False)
                 .head(cfg.TOP_N_STATES))
print(state_yield.round(3))

fig, ax = plt.subplots(figsize=(10, 6))
state_yield.sort_values().plot.barh(ax=ax, color="#b7e4c7", edgecolor="white")
ax.set_title(f"Top {cfg.TOP_N_STATES} States by Average Yield", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean Yield")
plt.tight_layout()
save_fig(fig, "04_state_yield", subdir="eda")

# ─── EDA 6: Year-over-Year Trend ──────────────────────────────────────────────
print("\n[EDA-6] Year-over-Year Yield Trend")
yearly = df.groupby("Crop_Year")[cfg.TARGET_COL].agg(["mean", "median"]).reset_index()
print(yearly.tail(10))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(yearly["Crop_Year"], yearly["mean"],   marker="o", label="Mean Yield",   color="#1b4332", lw=2)
ax.plot(yearly["Crop_Year"], yearly["median"], marker="s", label="Median Yield", color="#52b788", lw=2, linestyle="--")
ax.set_title("Year-over-Year Crop Yield Trend (1997–2020)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Yield")
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
plt.tight_layout()
save_fig(fig, "05_yearly_trend", subdir="eda")

# ─── EDA 7: Correlation Heatmap ───────────────────────────────────────────────
print("\n[EDA-7] Correlation Heatmap")
num_features = [
    "log_Yield", "log_Area", "log_Production", "Annual_Rainfall",
    "log_Fertilizer", "log_Pesticide",
    "Fertilizer_per_Area", "Pesticide_per_Area",
    "Fertilizer_x_Rainfall", "StateCrop_AvgYield"
]
num_features = [c for c in num_features if c in df.columns]
corr_matrix = df[num_features].corr()
print(corr_matrix["log_Yield"].sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, square=True,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "06_correlation_heatmap", subdir="eda")

# ─── EDA 8: Rainfall vs Yield Scatter ────────────────────────────────────────
print("\n[EDA-8] Rainfall vs Yield")
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(df["Annual_Rainfall"], df["log_Yield"],
                alpha=0.2, s=8, c=df["Season_Code"], cmap="tab10")
plt.colorbar(sc, ax=ax, label="Season Code")
ax.set_title("Annual Rainfall vs log(Yield) — Coloured by Season", fontsize=13, fontweight="bold")
ax.set_xlabel("Annual Rainfall (mm)")
ax.set_ylabel("log(Yield + 1)")
plt.tight_layout()
save_fig(fig, "07_rainfall_vs_yield", subdir="eda")

# ─── EDA 9: Fertilizer vs Yield ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df["log_Fertilizer"], df["log_Yield"],
           alpha=0.2, s=8, color="#40916c")
ax.set_title("log(Fertilizer) vs log(Yield)", fontsize=13, fontweight="bold")
ax.set_xlabel("log(Fertilizer + 1)")
ax.set_ylabel("log(Yield + 1)")
plt.tight_layout()
save_fig(fig, "08_fertilizer_vs_yield", subdir="eda")

# ─── EDA 10: Yield Heatmap — State × Season ──────────────────────────────────
print("\n[EDA-10] State × Season Yield Heatmap")
pivot = df.pivot_table(
    values=cfg.TARGET_COL, index="State", columns="Season",
    aggfunc="mean"
).fillna(0)
print(pivot.head())

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(pivot, cmap="YlGn", annot=True, fmt=".1f",
            linewidths=0.3, ax=ax, cbar_kws={"label": "Mean Yield"})
ax.set_title("Average Yield — State × Season Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "09_state_season_heatmap", subdir="eda")

# ─── EDA 11: Rainfall Zone Distribution ──────────────────────────────────────
print("\n[EDA-11] Rainfall Zone Analysis")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

zone_counts = df["Rainfall_Zone"].value_counts()
axes[0].bar(zone_counts.index, zone_counts.values, color="#95d5b2", edgecolor="white")
axes[0].set_title("Records per Rainfall Zone")
axes[0].set_xlabel("Zone")
axes[0].set_ylabel("Count")

zone_yield = df.groupby("Rainfall_Zone")[cfg.TARGET_COL].median()
axes[1].bar(zone_yield.index, zone_yield.values, color="#52b788", edgecolor="white")
axes[1].set_title("Median Yield per Rainfall Zone")
axes[1].set_xlabel("Zone")
axes[1].set_ylabel("Median Yield")

plt.suptitle("Rainfall Zone Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "10_rainfall_zone_analysis", subdir="eda")

# ─── EDA 12: Crop Count Distribution ─────────────────────────────────────────
crop_counts = df["Crop"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 7))
crop_counts.sort_values().plot.barh(ax=ax, color="#b7e4c7", edgecolor="white")
ax.set_title("Top 20 Crops by Record Count", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Records")
plt.tight_layout()
save_fig(fig, "11_crop_count_distribution", subdir="eda")

# ─── EDA 13: Era-wise Trend ───────────────────────────────────────────────────
if "Era" in df.columns:
    era_stats = df.groupby("Era")[cfg.TARGET_COL].agg(["mean","median","count"])
    print("\n[EDA-13] Era-Wise Statistics:")
    print(era_stats.round(3))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(era_stats.index.astype(str), era_stats["mean"], color=["#95d5b2","#52b788","#1b4332"],
           edgecolor="white")
    ax.set_title("Mean Yield by Era", fontsize=13, fontweight="bold")
    ax.set_xlabel("Era")
    ax.set_ylabel("Mean Yield")
    plt.tight_layout()
    save_fig(fig, "12_era_yield", subdir="eda")

print(f"\n{'='*60}")
print(f"  EDA COMPLETE — All plots saved to reports/eda/")
print(f"{'='*60}")
