"""SARIMA baseline model for electricity demand forecasting.

Uses daily-averaged demand to make SARIMA tractable.
Model: SARIMA(1,1,1)(1,1,1,7) — AR, differencing, MA with weekly seasonality.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluate import ModelEvaluator
from src.features import TARGET_COL, engineer_features, load_data
from src.utils import MODELS_DIR, OUTPUTS_DIR, save_json

TRAIN_END = "2019-12-31"
TEST_START = "2020-01-01"


def train() -> dict:
    """Run the full SARIMA training and evaluation pipeline.

    Steps:
        1. Load data and resample to daily averages
        2. Temporal split (train: <2020, test: 2020)
        3. Fit SARIMA(1,1,1)(1,1,1,7)
        4. Forecast test period
        5. Compute and save metrics

    Returns:
        Metrics dictionary.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load and aggregate to daily ---
    df = load_data()
    daily_demand: pd.Series = df[TARGET_COL].resample("D").mean()
    daily_demand = daily_demand.dropna()

    print(f"Daily demand series: {len(daily_demand)} days "
          f"({daily_demand.index.min()} to {daily_demand.index.max()})")

    # --- Temporal split ---
    train_daily = daily_demand[daily_demand.index <= TRAIN_END]
    test_daily = daily_demand[daily_demand.index >= TEST_START]

    print(f"Train: {len(train_daily)} days  |  Test: {len(test_daily)} days")

    # --- Fit SARIMA ---
    print("\nFitting SARIMA(1,1,1)(1,1,1,7) — this may take a few minutes...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_sarima = SARIMAX(
            train_daily,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model_sarima.fit(disp=False)

    print(results.summary())
    print(f"\nAIC: {results.aic:.2f}  |  BIC: {results.bic:.2f}")

    # Save model
    results.save(str(MODELS_DIR / "sarima_model.pkl"))
    print(f"SARIMA model saved to {MODELS_DIR / 'sarima_model.pkl'}")

    # --- Forecast test period ---
    n_test = len(test_daily)
    forecast = results.forecast(steps=n_test)
    forecast.index = test_daily.index

    # Align lengths
    actual = test_daily.values
    predicted = forecast.values

    # Save predictions for comparison plots
    np.save(OUTPUTS_DIR / "sarima_predictions.npy", predicted)
    np.save(OUTPUTS_DIR / "sarima_actuals.npy", actual)
    pd.Series(forecast.values, index=test_daily.index).to_csv(
        OUTPUTS_DIR / "sarima_forecast.csv", header=["sarima_forecast"]
    )

    # --- Evaluate ---
    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(actual, predicted, "SARIMA")

    save_json(metrics, OUTPUTS_DIR / "sarima_metrics.json")
    print(f"\nSARIMA metrics saved to {OUTPUTS_DIR / 'sarima_metrics.json'}")

    return metrics


if __name__ == "__main__":
    train()
