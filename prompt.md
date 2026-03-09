Read this prompt.md and build an upgraded version of my
Electricity Demand Forecasting project.

## Context

I have an existing project with LSTM + SARSA for electricity
demand forecasting. The dataset is continuous_dataset.csv with
48,048 hourly records from 2015-2020, 17 columns including
weather data for 3 cities, holiday flags, and school day flags.
Target variable: nat_demand (national electricity demand in MW).

The existing project has these critical problems to fix:

1. LSTM predictions in comparison plots used random numbers
2. SARSA was not doing real forecasting — just tweaking actuals
3. Metrics were computed on synthetic random data
4. Only 6 of 17 available features were used
5. No MAPE reported — just MSE and MAE
6. No temporal train/test split (used random split on time series)
7. No Streamlit deployment
8. No model saving

## Project Structure to Create

electricity-demand-forecasting/
├── data/
│ └── continuous_dataset.csv
├── notebooks/
│ └── 01_forecasting.ipynb # clean EDA + modeling notebook
├── src/
│ ├── features.py # feature engineering
│ ├── train_lstm.py # LSTM training
│ ├── train_sarima.py # SARIMA model
│ ├── evaluate.py # metrics and evaluation
│ └── utils.py # helpers
├── models/
│ └── .gitkeep
├── outputs/
│ └── .gitkeep
├── app/
│ └── streamlit_app.py
├── requirements.txt
└── README.md

## Tech Stack

Python 3.10+, pandas, numpy, tensorflow/keras,
statsmodels (SARIMA), scikit-learn, matplotlib,
seaborn, plotly, streamlit, joblib

## Step-by-Step Instructions

### Step 1: Data Loading & Feature Engineering (src/features.py)

Load continuous_dataset.csv.
Parse datetime as index.

Use ALL relevant columns:
features = [
'T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc',
'T2M_san', 'QV2M_san', 'TQL_san', 'W2M_san',
'T2M_dav', 'QV2M_dav', 'TQL_dav', 'W2M_dav',
'holiday', 'school', 'Holiday_ID'
]
target = 'nat_demand'

Engineer these additional time features:

- hour: 0-23 from datetime index
- day_of_week: 0-6
- month: 1-12
- is_weekend: binary (day_of_week >= 5)
- hour_sin: np.sin(2 _ np.pi _ hour / 24) # cyclical encoding
- hour_cos: np.cos(2 _ np.pi _ hour / 24)
- month_sin: np.sin(2 _ np.pi _ month / 12)
- month_cos: np.cos(2 _ np.pi _ month / 12)

Add lag features on nat_demand:

- lag_24: demand 24 hours ago (same hour yesterday)
- lag_168: demand 168 hours ago (same hour last week)
- rolling_mean_24: 24-hour rolling mean
- rolling_std_24: 24-hour rolling std

Drop rows with NaN from lag features.

CRITICAL: Use temporal train/test split, NOT random split.
Train: 2015-01-01 to 2019-12-31
Test: 2020-01-01 to 2020-06-27
Print train size, test size, split date.

### Step 2: LSTM Model (src/train_lstm.py)

Use a proper sequence-to-point forecasting setup.

Sequence length: 168 (one week of hourly data)
Forecast horizon: 24 (predict next 24 hours)

create_sequences(data, seq_len=168, horizon=24):

- X: sequences of seq_len timesteps
- y: next 24 hours of nat_demand
- This is multi-step forecasting — much stronger than
  single-step

Scale features with MinMaxScaler.
CRITICAL: fit scaler on train only, transform both train and test.
Save scaler as models/feature_scaler.pkl
Save separate target scaler as models/target_scaler.pkl

Architecture:
model = Sequential([
Input(shape=(168, num_features)),
LSTM(64, return_sequences=True),
Dropout(0.2),
LSTM(32, return_sequences=False),
Dropout(0.2),
Dense(24) # predict 24 hours ahead
])
model.compile(optimizer=Adam(lr=0.001), loss='mse',
metrics=['mae'])

Callbacks:

- EarlyStopping(patience=10, restore_best_weights=True)
- ReduceLROnPlateau(patience=5, factor=0.5)
- ModelCheckpoint saving best model to models/lstm_best.keras

Train: epochs=50, batch_size=64, validation_split=0.1
(validation split from END of training data only)

After training compute and print:

- MAE on test set (in original MW scale)
- RMSE on test set
- MAPE on test set: mean(|actual-pred|/actual) \* 100
- R² score

Save all metrics to outputs/lstm_metrics.json

### Step 3: SARIMA Model (src/train_sarima.py)

Replace the flawed SARSA with proper SARIMA.
SARIMA is the correct statistical baseline for this data.

Use hourly data aggregated to daily averages to make
SARIMA tractable:
daily_demand = df['nat_demand'].resample('D').mean()

Split same way: train before 2020, test 2020.

Fit SARIMA(1,1,1)(1,1,1,7):
from statsmodels.tsa.statespace.sarimax import SARIMAX
model_sarima = SARIMAX(
train_daily,
order=(1,1,1),
seasonal_order=(1,1,1,7),
enforce_stationarity=False,
enforce_invertibility=False
)
results = model_sarima.fit(disp=False)
results.save('models/sarima_model.pkl')

Generate forecast for test period.
Compute MAE, RMSE, MAPE on daily test set.
Save metrics to outputs/sarima_metrics.json

Print model summary and AIC/BIC scores.

### Step 4: Evaluation & Comparison (src/evaluate.py)

Build a ModelEvaluator class:

compute_metrics(actual, predicted, model_name) -> dict
Compute and return:

- MAE
- RMSE
- MAPE: mean(abs(actual-pred)/actual) \* 100
- R²
- Max error
- % of predictions within 5% of actual
- % of predictions within 10% of actual

compare_models(lstm_results, sarima_results):
Print a side-by-side comparison table:
| Metric | LSTM | SARIMA |
|---------------|---------|---------|
| MAE (MW) | X | X |
| RMSE (MW) | X | X |
| MAPE (%) | X | X |
| R² | X | X |
| Within 5% | X% | X% |
| Within 10% | X% | X% |

Save comparison to outputs/model_comparison.json

plot_predictions(actual, lstm_pred, sarima_pred):

- Figure 1: Full test period actual vs both models
- Figure 2: Zoom into first 2 weeks of test period
- Figure 3: Scatter plot actual vs predicted (both models)
- Figure 4: Residuals over time (both models)
- Figure 5: Error distribution histogram
  Save all to outputs/ as PNGs.

### Step 5: EDA Notebook (notebooks/01_forecasting.ipynb)

Section 1: Dataset Overview

- Shape, date range, missing values
- Target distribution (histogram + stats)
- Time series plot of full demand history

Section 2: Temporal Patterns

- Average demand by hour of day (line chart)
- Average demand by day of week (bar chart)
- Average demand by month (bar chart)
- Heatmap: hour vs day_of_week average demand
  (this is the most visually impressive plot)

Section 3: Feature Correlations

- Correlation heatmap of all features vs nat_demand
- Scatter plots: temperature (all 3 cities) vs demand
- Holiday vs non-holiday demand distribution (boxplot)
- School day vs non-school day demand (boxplot)

Section 4: Stationarity Analysis

- Plot ACF and PACF of nat_demand
- Augmented Dickey-Fuller test result
- Seasonal decomposition plot (trend, seasonal, residual)

Section 5: Model Results

- Import and display all metrics from outputs/
- Show all comparison plots from evaluate.py
- Write 3-5 key findings as markdown cells

### Step 6: Streamlit App (app/streamlit_app.py)

3-tab dashboard:

Tab 1 — "Data Explorer"

- Date range selector (slider)
- Line chart of nat_demand over selected period
- Key stats: min, max, mean, std demand
- Toggle: overlay temperature data on secondary y-axis

Tab 2 — "Forecast"

- Date picker: select a start date in test period
- Slider: forecast horizon (24, 48, or 168 hours)
- "Generate Forecast" button
- Load saved LSTM model and scalers
- Show: actual vs LSTM forecast chart (Plotly)
- Show metrics for that specific forecast window
- Show: which features were most important (bar chart
  using permutation importance — precompute and save)

Tab 3 — "Model Comparison"

- Show side-by-side metrics table (from model_comparison.json)
- LSTM vs SARIMA comparison charts
- Error distribution comparison
- Key insight callout box:
  "LSTM achieves X% MAPE — equivalent to
  predicting X MW error on average daily demand of Y MW"
  This is the business translation recruiters love

### Step 7: README.md (rewrite completely)

1. Project Overview (2 sentences, include MAPE result)
2. Business Problem
3. Dataset (48K hourly records, 2015-2020, Panama)
4. Approach: why LSTM for short-term, SARIMA as baseline
5. Feature Engineering (list all 20+ features used)
6. Results Table (fill after running):
   | Model | MAE | RMSE | MAPE | R² |
   | LSTM | | | | |
   | SARIMA | | | | |
7. Key Findings (3 bullets from EDA + model analysis)
8. How to Run
9. Streamlit demo (add link if deployed)
10. Future Work:
    - Transformer-based model (Temporal Fusion Transformer)
    - Multi-step probabilistic forecasting with prediction intervals
    - Real-time data ingestion pipeline

## Code Quality

- Type hints and docstrings on all functions
- pathlib.Path throughout
- requirements.txt with pinned versions
- Notebook runs top-to-bottom without errors
- No random data used anywhere for metrics

## Resume Bullets to Aim For After Building

- "Rebuilt LSTM forecasting model on 48K hourly electricity
  records (2015-2020); engineered 20+ temporal and weather
  features; achieved X% MAPE on 6-month holdout —
  improvement over SARIMA baseline (Y% MAPE)"
- "Built 24-hour multi-step forecast pipeline with proper
  temporal train/test split, separate scalers, and model
  checkpointing; deployed interactive Streamlit dashboard
  with real-time forecast generation"

## Start

Create the full project structure first.
Then confirm before writing any logic.
Do not use random data anywhere for metrics or plots.
All evaluation must use real model predictions on the
held-out test set (2020 data).
