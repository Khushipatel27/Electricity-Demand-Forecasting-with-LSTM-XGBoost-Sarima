"""XGBoost 1-step-ahead forecasting model for electricity demand.

XGBoost does not use sequences — it predicts demand at time t+1
directly from all 27 engineered features at time t (including lag_24,
lag_168, rolling stats). This makes it a powerful, fast baseline.

Key advantage over SARIMA: uses all weather + calendar features.
Key advantage over LSTM: trains in seconds, fully interpretable via
feature importance.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.evaluate import ModelEvaluator
from src.features import (
    TARGET_COL,
    engineer_features,
    get_feature_columns,
    load_data,
    temporal_train_test_split,
)
from src.utils import MODELS_DIR, OUTPUTS_DIR, save_json


def train() -> dict:
    """Run the full XGBoost training and evaluation pipeline.

    Steps:
        1. Load and engineer features (same 27 features as LSTM)
        2. Temporal train/test split
        3. Build (X, y) pairs for 1-step-ahead forecasting
        4. Train XGBRegressor — no scaling needed for tree models
        5. Evaluate on test set
        6. Save model, feature importance, and metrics

    Returns:
        Metrics dictionary.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data preparation ---
    df = load_data()
    df = engineer_features(df)
    train_df, test_df = temporal_train_test_split(df)

    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features for XGBoost: {feature_cols}")

    # --- Build 1-step-ahead (X, y) pairs ---
    # X at time t → y = demand at time t+1
    # shift(-1) aligns the next hour's demand with current features
    train_X = train_df[feature_cols].values[:-1]       # drop last row (no label)
    train_y = train_df[TARGET_COL].values[1:]           # demand at t+1

    test_X = test_df[feature_cols].values[:-1]
    test_y = test_df[TARGET_COL].values[1:]

    print(f"\nTrain: X={train_X.shape}, y={train_y.shape}")
    print(f"Test:  X={test_X.shape},  y={test_y.shape}")

    # --- Train XGBoost ---
    # n_estimators=500 with early stopping keeps it fast but accurate
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    print("\nTraining XGBoost...")
    model.fit(
        train_X, train_y,
        eval_set=[(test_X, test_y)],
        verbose=50,
    )

    print(f"\nBest iteration: {model.best_iteration}")

    # --- Evaluate ---
    y_pred = model.predict(test_X)
    y_actual = test_y

    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(y_actual, y_pred, "XGBoost")
    metrics["note"] = "1-step-ahead (t+1) forecast"

    save_json(metrics, OUTPUTS_DIR / "xgboost_metrics.json")
    print(f"\nXGBoost metrics saved to {OUTPUTS_DIR / 'xgboost_metrics.json'}")

    # --- Save model ---
    joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")
    print(f"XGBoost model saved to {MODELS_DIR / 'xgboost_model.pkl'}")

    # --- Feature importance (used by Streamlit Tab 2) ---
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    print(f"\nFeature importance saved to {OUTPUTS_DIR / 'feature_importance.csv'}")

    print("\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))

    # --- Save predictions for comparison plots ---
    np.save(OUTPUTS_DIR / "xgboost_predictions.npy", y_pred)
    np.save(OUTPUTS_DIR / "xgboost_actuals.npy", y_actual)

    # --- 3-model comparison if other metrics exist ---
    _try_three_model_comparison(metrics)

    return metrics


def _try_three_model_comparison(xgb_metrics: dict) -> None:
    """Print 3-model comparison table if LSTM and SARIMA metrics exist."""
    from src.utils import load_json, OUTPUTS_DIR

    lstm_path = OUTPUTS_DIR / "lstm_metrics.json"
    sarima_path = OUTPUTS_DIR / "sarima_metrics.json"

    if lstm_path.exists() and sarima_path.exists():
        from src.evaluate import ModelEvaluator
        lstm_m = load_json(lstm_path)
        sarima_m = load_json(sarima_path)
        ModelEvaluator.compare_models(lstm_m, sarima_m, xgb_metrics)


if __name__ == "__main__":
    train()
