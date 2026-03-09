"""Generate model comparison plots and 3-model metrics table.

Run after training all three models:
    python -m src.train_lstm
    python -m src.train_xgboost
    python -m src.train_sarima

Then run this script:
    python -m src.compare_models
"""

import numpy as np
import pandas as pd

from src.evaluate import ModelEvaluator, plot_predictions
from src.features import TARGET_COL, engineer_features, load_data, temporal_train_test_split
from src.utils import OUTPUTS_DIR, load_json


def main() -> None:
    # ── Load saved metrics ──────────────────────────────────────────────────
    lstm_m   = load_json(OUTPUTS_DIR / "lstm_metrics.json")   if (OUTPUTS_DIR / "lstm_metrics.json").exists()   else None
    sarima_m = load_json(OUTPUTS_DIR / "sarima_metrics.json") if (OUTPUTS_DIR / "sarima_metrics.json").exists() else None
    xgb_m    = load_json(OUTPUTS_DIR / "xgboost_metrics.json") if (OUTPUTS_DIR / "xgboost_metrics.json").exists() else None

    # ── Print comparison table ──────────────────────────────────────────────
    if lstm_m and sarima_m:
        print("\n=== Model Comparison ===")
        ModelEvaluator.compare_models(lstm_m, sarima_m, xgb_m)

    # ── Load predictions for plots ──────────────────────────────────────────
    # LSTM predictions (shape: n_samples x 24) — use horizon 0 (1-step) for alignment
    lstm_pred_path   = OUTPUTS_DIR / "lstm_predictions.npy"
    lstm_actual_path = OUTPUTS_DIR / "lstm_actuals.npy"
    sarima_pred_path = OUTPUTS_DIR / "sarima_predictions.npy"
    sarima_actual_path = OUTPUTS_DIR / "sarima_actuals.npy"
    xgb_pred_path    = OUTPUTS_DIR / "xgboost_predictions.npy"
    xgb_actual_path  = OUTPUTS_DIR / "xgboost_actuals.npy"

    has_lstm   = lstm_pred_path.exists() and lstm_actual_path.exists()
    has_sarima = sarima_pred_path.exists() and sarima_actual_path.exists()
    has_xgb    = xgb_pred_path.exists() and xgb_actual_path.exists()

    if not has_lstm:
        print("LSTM predictions not found — run python -m src.train_lstm first.")
        return

    # Load LSTM preds — shape (n, 24). Use hour +1 slice for hourly alignment.
    lstm_pred_2d = np.load(lstm_pred_path)     # (n_samples, 24)
    lstm_act_2d  = np.load(lstm_actual_path)   # (n_samples, 24)
    lstm_pred_1h = lstm_pred_2d[:, 0]          # first horizon step for alignment
    lstm_act_1h  = lstm_act_2d[:, 0]

    # For SARIMA (daily) — upsample to match LSTM hourly length approximately
    if has_sarima:
        sarima_pred_daily = np.load(sarima_pred_path)
        sarima_act_daily  = np.load(sarima_actual_path)
        # Repeat each daily value 24 times to get hourly-aligned array
        sarima_pred_hourly = np.repeat(sarima_pred_daily, 24)[: len(lstm_act_1h)]
        sarima_act_hourly  = np.repeat(sarima_act_daily, 24)[: len(lstm_act_1h)]
        # Fill to same length
        n = len(lstm_act_1h)
        if len(sarima_pred_hourly) < n:
            sarima_pred_hourly = np.pad(sarima_pred_hourly, (0, n - len(sarima_pred_hourly)), mode="edge")
    else:
        print("SARIMA predictions not found — plots will use LSTM only.")
        sarima_pred_hourly = lstm_pred_1h.copy()  # fallback placeholder

    # XGBoost (already hourly)
    xgb_pred_hourly = None
    if has_xgb:
        xgb_pred = np.load(xgb_pred_path)
        xgb_pred_hourly = xgb_pred[: len(lstm_act_1h)]

    # Build DatetimeIndex for x-axis
    df = load_data()
    df = engineer_features(df)
    _, test_df = temporal_train_test_split(df)
    test_index = test_df.index[: len(lstm_act_1h)]

    # ── Generate all 5 plots ────────────────────────────────────────────────
    print("\nGenerating comparison plots...")
    plot_predictions(
        actual=lstm_act_1h,
        lstm_pred=lstm_pred_1h,
        sarima_pred=sarima_pred_hourly,
        xgboost_pred=xgb_pred_hourly,
        index=test_index,
    )
    print("Done. Check outputs/ directory.")


if __name__ == "__main__":
    main()
