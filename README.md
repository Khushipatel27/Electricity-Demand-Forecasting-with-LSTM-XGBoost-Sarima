# ⚡ Electricity Demand Forecasting — Panama (2015–2020)

> Predicting national electricity demand using deep learning, gradient boosting, and statistical models — trained on 48,000+ hourly records from Panama's national grid.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-orange?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

---

## 💡 Why This Project?

Electricity demand forecasting is critical infrastructure. Grid operators need accurate forecasts to:

- **Prevent blackouts** — under-supply disrupts hospitals, industry, and homes
- **Reduce costs** — over-generation wastes fuel and money
- **Enable renewables** — solar/wind integration requires precise load prediction

This project builds a complete end-to-end forecasting pipeline with three models, 27 engineered features, and a live interactive dashboard.

---

## 📊 Results at a Glance

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | R² | Horizon |
|-------|----------|-----------|----------|----|---------|
| **LSTM** | 81.67 | 112.48 | **7.15%** | 0.59 | 24-hour multi-step |
| **XGBoost** | 58.74 | 81.19  | 5.13% | 0.78 | 1-step ahead |
| **SARIMA** | 127.13 | 152.26 | 11.47% | -0.65 | Daily baseline |

> **LSTM achieves 7.15% MAPE** — predicting demand within ~82 MW on an average load of ~1,150 MW.
> SARIMA serves as a statistical baseline. XGBoost metrics update after running `python -m src.train_xgboost`.

---

## 🗂️ Dataset

| Property | Detail |
|----------|--------|
| Source | Panama national electricity grid |
| Records | 48,048 hourly observations |
| Period | Jan 2015 — Jun 2020 |
| Features | 17 raw columns |
| Target | `nat_demand` — national demand in MW |
| Cities | Tocumen · Santiago · David (weather data for each) |

**Train / Test Split (temporal — no data leakage)**
```
Train │ 2015-01-10 ──────────────────────── 2019-12-31 │ 43,607 rows
Test  │ 2020-01-01 ────────── 2020-06-27 │  4,273 rows
```

---

## 🧠 Models

### LSTM — Deep Learning (Primary)
- **Input**: 168-hour sequence (1 full week) → **Output**: next 24 hours
- Architecture: `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(24)`
- Trained with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Scalers fit on training data only — no leakage

### XGBoost — Gradient Boosting
- 1-step-ahead forecast using all 27 features at time `t` → demand at `t+1`
- Trains in ~30 seconds vs 30 min for LSTM
- Provides feature importance rankings

### SARIMA(1,1,1)(1,1,1,7) — Statistical Baseline
- Fitted on daily-averaged demand with weekly seasonality
- Interpretable, no feature engineering needed
- Useful for benchmarking and understanding baseline difficulty

---

## 🔧 Feature Engineering

**27 features** engineered from 17 raw columns:

| Category | Features |
|----------|----------|
| **Weather** (12) | Temperature, humidity, liquid water, wind speed for Tocumen, Santiago, David |
| **Calendar** (3) | `holiday`, `school`, `Holiday_ID` |
| **Time** (4) | `hour`, `day_of_week`, `month`, `is_weekend` |
| **Cyclical** (4) | `hour_sin/cos`, `month_sin/cos` — prevent hour 0 ≠ hour 23 issue |
| **Lag** (4) | `lag_24` (yesterday same hour), `lag_168` (last week), `rolling_mean_24`, `rolling_std_24` |

---

## 🔍 Key Findings

- **Daily peaks at 7–9am and 6–9pm** — demand follows human activity patterns with strong weekly seasonality
- **Temperature has a U-shaped relationship** with demand — both hot and cool weather drive higher consumption (cooling + heating loads)
- **Holidays drop demand by ~8–12%** vs equivalent weekdays — human scheduling is a significant driver
- **Lag features dominate importance** — yesterday's same-hour demand is the single strongest predictor, followed by temperature

---

## 📁 Project Structure

```
⚡ electricity-demand-forecasting/
│
├── 📂 data/
│   └── continuous_dataset.csv        # dataset (place here)
│
├── 📂 src/
│   ├── features.py                   # data loading + 27-feature engineering
│   ├── train_lstm.py                 # LSTM training (seq2point, 24-step)
│   ├── train_xgboost.py              # XGBoost training + feature importance
│   ├── train_sarima.py               # SARIMA(1,1,1)(1,1,1,7) baseline
│   ├── evaluate.py                   # ModelEvaluator + 5 comparison plots
│   ├── compare_models.py             # generate plots + 3-model table
│   └── utils.py                      # shared paths and helpers
│
├── 📂 notebooks/
│   └── 01_forecasting.ipynb          # EDA + stationarity + model results
│
├── 📂 app/
│   └── streamlit_app.py              # 4-tab interactive dashboard
│
├── 📂 models/                        # saved .keras + .pkl files
├── 📂 outputs/                       # metrics JSON + PNG plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Khushipatel27/Electricity-Demand-Forecasting-with-LSTM-XGBoost-Sarima.git
cd Electricity-Demand-Forecasting-with-LSTM-XGBoost-Sarima
pip install -r requirements.txt
```

### 2. Add the dataset

Place `continuous_dataset.csv` in the `data/` folder.
_(The code will also find it automatically if left in the project root.)_

### 3. Train the models

```bash
# LSTM — deep learning (24-hour multi-step)
python -m src.train_lstm

# XGBoost — gradient boosting (fast, ~30 seconds)
python -m src.train_xgboost

# SARIMA — statistical baseline
python -m src.train_sarima

# Generate comparison plots + metrics table
python -m src.compare_models
```

### 4. Explore the notebook

```bash
jupyter notebook notebooks/01_forecasting.ipynb
```

### 5. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🖥️ Streamlit Dashboard

4-tab interactive dashboard:

| Tab | What you can do |
|-----|----------------|
| 📊 **Overview** | KPI cards, full demand history, hour×weekday heatmap |
| 🔍 **Data Explorer** | Zoom into any date range, overlay weather, see holiday patterns |
| 🔮 **Live Forecast** | Pick any date in 2020, run LSTM or XGBoost, see per-hour error breakdown |
| 📈 **Model Comparison** | Side-by-side metrics, interactive charts, feature importance |

> **Live Demo**: _Deploy link coming soon_

---

## 🔭 Future Work

- **Temporal Fusion Transformer (TFT)** — attention-based model built for multi-horizon time series with built-in interpretability
- **Probabilistic forecasting** — prediction intervals via conformal prediction for risk-aware grid planning
- **Real-time pipeline** — connect to Panama's ETESA API for live inference and automated retraining

---

## 🛠️ Tech Stack

`Python 3.10` · `TensorFlow / Keras` · `XGBoost` · `statsmodels` · `scikit-learn` · `pandas` · `plotly` · `Streamlit` · `joblib`
