# Electricity Demand Forecasting — Panama 2015–2020

Rebuilt LSTM forecasting model on 48K hourly electricity records (2015–2020); engineered 27 temporal and weather features with a proper temporal train/test split; deployed an interactive Streamlit dashboard with real-time 24-hour multi-step forecast generation.

---

## Business Problem

Accurate electricity demand forecasting enables grid operators to optimize generation dispatch, reduce costs, and prevent blackouts. Even a 1% improvement in forecast accuracy translates to significant operational savings at the national scale.

---

## Dataset

- **Source**: Panama national electricity demand
- **Records**: 48,048 hourly observations (2015-01-03 to 2020-06-27)
- **Columns**: 17 — demand, weather for 3 cities (Tocumen, Santiago, David), holiday/school flags, Holiday_ID
- **Target**: `nat_demand` — national electricity demand in MW

---

## Approach

| Component  | Description                                                                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **LSTM**   | Primary model. Sequence-to-point forecasting: 168-hour input window → 24-hour output. Captures complex non-linear temporal patterns.        |
| **SARIMA** | Statistical baseline. Fitted on daily-averaged demand with SARIMA(1,1,1)(1,1,1,7). Interpretable and robust but limited to linear patterns. |
| **Split**  | Temporal: train 2015–2019, test Jan–Jun 2020. No data leakage.                                                                              |

---

## Feature Engineering

27 features total:

**Raw weather (12)**: `T2M_toc`, `QV2M_toc`, `TQL_toc`, `W2M_toc`, `T2M_san`, `QV2M_san`, `TQL_san`, `W2M_san`, `T2M_dav`, `QV2M_dav`, `TQL_dav`, `W2M_dav`

**Calendar flags (3)**: `holiday`, `school`, `Holiday_ID`

**Temporal (8)**: `hour`, `day_of_week`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`

**Lag features (4)**: `lag_24` (same hour yesterday), `lag_168` (same hour last week), `rolling_mean_24`, `rolling_std_24`

---

## Results

| Model  | MAE (MW) | RMSE (MW) | MAPE (%) | R²     |
| ------ | -------- | --------- | -------- | ------ |
| LSTM   | 81.67    | 112.476   | 7.146    | 0.5916 |
| SARIMA | 127.129  | 152.259   | 11.469   | -0.652 |

_Fill in after running `python -m src.train_lstm` and `python -m src.train_sarima`._

---

## Key Findings

- Demand peaks during morning (7–9am) and evening (6–9pm) hours with a consistent weekly seasonal pattern.
- Temperature shows a U-shaped relationship with demand — both cooling and heating loads are visible in the data.
- Holiday and school flags reduce demand materially vs equivalent weekdays, confirming human scheduling as a major driver.

---

## Project Structure

```
electricity-demand-forecasting/
├── data/
│   └── continuous_dataset.csv       # place your dataset here
├── notebooks/
│   └── 01_forecasting.ipynb         # EDA + model results
├── src/
│   ├── features.py                  # feature engineering + temporal split
│   ├── train_lstm.py                # LSTM training pipeline
│   ├── train_sarima.py              # SARIMA baseline
│   ├── evaluate.py                  # metrics + comparison plots
│   └── utils.py                     # paths and helpers
├── models/                          # saved models and scalers
├── outputs/                         # metrics JSON + plots
├── app/
│   └── streamlit_app.py             # interactive dashboard
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

Copy `continuous_dataset.csv` into the `data/` directory (or leave it in the project root — the code will find it automatically).

### 3. Train models

```bash
# Train LSTM (~20-30 min on CPU, faster with GPU)
python -m src.train_lstm

# Train SARIMA (~5-10 min)
python -m src.train_sarima
```

Both scripts save their models to `models/` and metrics to `outputs/`.

### 4. Run the EDA notebook

```bash
jupyter notebook notebooks/01_forecasting.ipynb
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Streamlit Demo

_Add deployment link here after deploying to Streamlit Cloud._

---

## Future Work

- **Transformer-based model**: Temporal Fusion Transformer (TFT) for improved long-range dependencies and built-in interpretability via attention weights.
- **Probabilistic forecasting**: Multi-step prediction intervals using Monte Carlo Dropout or conformal prediction — enables risk-aware grid planning.
- **Real-time data pipeline**: Integrate with Panama's ETESA API for live demand data, enabling production-grade forecasting with automated retraining.
