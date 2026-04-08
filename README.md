# 🌾 Crop Yield Prediction & Business Dashboarding

> End-to-end machine learning pipeline on 10+ years of Indian agricultural data  
> **Best Model: XGBoost | R² = 0.98 | CV R² = 0.982 ± 0.004**

---

## 📁 Project Structure

```
crop_yield_project/
├── data/
│   ├── crop_yield.csv               ← Raw dataset (19,689 rows)
│   ├── crop_yield_cleaned.csv       ← After cleaning (19,185 rows)
│   └── crop_yield_featured.csv      ← After feature engineering (30 cols)
│
├── outputs/
│   ├── best_model.pkl               ← Trained XGBoost model
│   ├── scaler.pkl                   ← StandardScaler for Ridge
│   ├── model_metrics.csv            ← Model comparison results
│   ├── test_predictions.csv         ← Predictions on holdout test set
│   └── crop_yield_powerbi.csv       ← Power BI ready export
│
├── reports/
│   ├── cleaning/                    ← Yield distribution plots
│   ├── eda/                         ← 12 EDA visualizations
│   └── model/                       ← Actual vs predicted, residuals, feature importance
│
├── config.py                        ← Centralized configuration
├── utils.py                         ← Shared utility functions
├── 01_data_cleaning.py              ← Cleaning + feature engineering
├── 02_eda.py                        ← Exploratory data analysis
├── 03_regression_model.py           ← Model training & evaluation
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Attribute        | Value                                      |
|------------------|--------------------------------------------|
| Source           | India Agriculture & Climate Dataset (IFPRI)|
| Rows (raw)       | 19,689                                     |
| Rows (cleaned)   | 19,185                                     |
| Features (final) | 30 columns                                 |
| Years            | 1997 – 2020 (24 years)                     |
| Crops            | 55 unique crops                            |
| States           | 30 Indian states                           |
| Seasons          | 6 (Kharif, Rabi, Whole Year, Autumn, Summer, Winter) |

**Columns:**
`Crop`, `Crop_Year`, `Season`, `State`, `Area`, `Production`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`, `Yield`

---

## 🔧 Setup & Installation

```bash
# 1. Clone / download the project folder
cd crop_yield_project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run scripts in order:

```bash
# Step 1 — Data Cleaning & Feature Engineering
python 01_data_cleaning.py

# Step 2 — Exploratory Data Analysis
python 02_eda.py

# Step 3 — Model Training & Evaluation
python 03_regression_model.py
```

Each script prints progress logs and saves outputs automatically.

---

## 🧹 Data Cleaning Steps

| Step | Action | Impact |
|------|--------|--------|
| Whitespace stripping | Strip Season, Crop, State columns | Prevents label mismatch errors |
| Zero Yield removal | Drop rows where Yield ≤ 0 | Removed 112 rows |
| Percentile capping | Yield clipped to [1st, 99th] pct | Removed 392 extreme outliers |
| IQR capping | Area, Production, Fertilizer, Pesticide capped at IQR × 3 | Reduced skew |
| Log transform | log1p on Area, Production, Fertilizer, Pesticide, Yield | Normalized distributions |

**Total rows removed: 504 (2.6%)** — minimal data loss with significant quality improvement.

---

## 🛠️ Feature Engineering (15+ Features)

| Feature | Description |
|---------|-------------|
| `log_Area` | log(Area+1) — reduces right skew |
| `log_Production` | log(Production+1) |
| `log_Fertilizer` | log(Fertilizer+1) |
| `log_Pesticide` | log(Pesticide+1) |
| `Fertilizer_per_Area` | Fertilizer usage intensity per ha |
| `Pesticide_per_Area` | Pesticide usage intensity per ha |
| `Production_per_Area` | Yield proxy (pre-feature) |
| `Fertilizer_to_Pesticide` | Input balance ratio |
| `Rainfall_Zone` | Binned: Very Low / Low / Moderate / High / Very High |
| `Season_Code` | Ordinal encoding of season |
| `Crop_Enc` | Label-encoded Crop |
| `State_Enc` | Label-encoded State |
| `Decade` | 1990s / 2000s / 2010s era grouping |
| `Era` | Three-period grouping: 1997-2004, 2005-2012, 2013-2020 |
| `Fertilizer_x_Rainfall` | Interaction: fertilizer × climate |
| `Area_x_Rainfall` | Interaction: scale × climate |
| `StateCrop_AvgYield` | Rolling historical mean yield per State-Crop pair |

---

## 🤖 Model Results

| Model | R² | MAE | RMSE | MAPE |
|-------|----|-----|------|------|
| **XGBoost** ← Best | **0.984** | **0.056** | **0.110** | **7.3%** |
| Random Forest | 0.973 | 0.060 | 0.141 | 7.7% |
| Ridge Regression | 0.913 | 0.151 | 0.254 | 21.6% |

> All metrics on log-transformed Yield. CV R² = 0.982 ± 0.004 (5-fold).  
> Original-scale R² = 0.960

### XGBoost Configuration
```python
n_estimators     = 500
learning_rate    = 0.05
max_depth        = 6
subsample        = 0.8
colsample_bytree = 0.8
min_child_weight = 3
reg_alpha        = 0.1
```

---

## 📈 Key EDA Findings

- **Whole Year** season has the highest average yield (12.97), driven by perennial crops
- **Sugarcane** and **Banana** are the highest-yielding crops (49 and 26 tons/ha)
- **Delhi**, **Kerala**, and **Tamil Nadu** lead in average state-level yield
- Yield shows a modest upward trend from 1997 to 2020
- `StateCrop_AvgYield` (historical crop-state mean) has the strongest correlation with Yield (r = 0.776)
- Moderate rainfall zones (500–1500 mm) produce the most consistent yields

---

## 📤 Power BI Export

The script generates `outputs/crop_yield_powerbi.csv` with:
- All original columns
- `Actual_Yield` and `Predicted_Yield`
- `Prediction_Error` and `Error_Pct`
- `Rainfall_Zone` and `Era` for slicing

See `POWERBI_GUIDE.md` for full dashboard setup instructions.

---

## 📌 Impact Summary

- Analysed **10+ years** of regional agricultural data (1997–2020)
- Processed **50,000+ data points** across 55 crop types and 6 seasonal variables
- Engineered **15+ features** — rainfall indices, soil/input ratios, seasonal dummies
- Built regression model achieving **R² = 0.98** (conservative resume claim: 0.88)
- Developed **4 Power BI dashboards** with region/season/crop drill-downs
- Identified high-yield crops (Sugarcane, Banana) and optimal states (Kerala, Tamil Nadu)
