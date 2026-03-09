"""LSTM multi-step forecasting model for electricity demand."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from src.evaluate import ModelEvaluator
from src.features import (
    TARGET_COL,
    engineer_features,
    get_feature_columns,
    load_data,
    temporal_train_test_split,
)
from src.utils import MODELS_DIR, OUTPUTS_DIR, save_json

# Sequence / forecast configuration
SEQ_LEN = 168    # one week of hourly data
HORIZON = 24     # predict next 24 hours


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) sequence pairs for multi-step forecasting.

    Args:
        features: 2D array of shape (n_timesteps, n_features).
        targets: 1D array of shape (n_timesteps,) — the target column.
        seq_len: Number of past timesteps used as input.
        horizon: Number of future timesteps to predict.

    Returns:
        X of shape (n_samples, seq_len, n_features),
        y of shape (n_samples, horizon).
    """
    X, y = [], []
    for i in range(len(features) - seq_len - horizon + 1):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len : i + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_model(num_features: int) -> "tf.keras.Model":
    """Build the LSTM architecture.

    Args:
        num_features: Number of input features per timestep.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(SEQ_LEN, num_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(HORIZON),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    model.summary()
    return model


def train() -> dict:
    """Run the full LSTM training pipeline.

    Steps:
        1. Load and engineer features
        2. Temporal train/test split
        3. Fit MinMaxScaler on train only
        4. Create sequences
        5. Train with callbacks
        6. Evaluate on test set
        7. Save model, scalers, and metrics

    Returns:
        Metrics dictionary.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data preparation ---
    df = load_data()
    df = engineer_features(df)
    train_df, test_df = temporal_train_test_split(df)

    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} feature columns: {feature_cols}")

    # Scale features: fit on train, transform both
    feature_scaler = MinMaxScaler()
    train_features_scaled = feature_scaler.fit_transform(train_df[feature_cols].values)
    test_features_scaled = feature_scaler.transform(test_df[feature_cols].values)

    # Scale target separately (for inverse transform of predictions)
    target_scaler = MinMaxScaler()
    train_target_scaled = target_scaler.fit_transform(
        train_df[[TARGET_COL]].values
    ).flatten()
    test_target_scaled = target_scaler.transform(
        test_df[[TARGET_COL]].values
    ).flatten()

    joblib.dump(feature_scaler, MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(target_scaler, MODELS_DIR / "target_scaler.pkl")
    print(f"Scalers saved to {MODELS_DIR}/")

    # --- Create sequences ---
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled)
    print(f"\nSequence shapes — X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"                  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # --- Build and train model ---
    model = build_model(num_features=len(feature_cols))

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        ModelCheckpoint(
            str(MODELS_DIR / "lstm_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate on test set ---
    y_pred_scaled = model.predict(X_test, verbose=0)  # (n_samples, 24)

    # Inverse-transform: reshape to (-1, 1) then back
    y_pred_mw = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(y_pred_scaled.shape)
    y_actual_mw = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).reshape(y_test.shape)

    # Flatten for overall metric computation
    y_pred_flat = y_pred_mw.flatten()
    y_actual_flat = y_actual_mw.flatten()

    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(y_actual_flat, y_pred_flat, "LSTM")

    # --- Per-horizon R² breakdown ---
    # Reveals true model quality: hour +1 is much more accurate than hour +24
    print("\nPer-horizon breakdown (LSTM — each row = predictions at that step ahead):")
    print(f"  {'Horizon':<12} {'MAPE':>8} {'R²':>8} {'MAE (MW)':>12}")
    print(f"  {'-'*44}")
    horizon_metrics = {}
    for h in [0, 2, 5, 11, 17, 23]:
        actual_h = y_actual_mw[:, h]
        pred_h = y_pred_mw[:, h]
        mape_h = float(np.mean(np.abs(actual_h - pred_h) / np.abs(actual_h)) * 100)
        r2_h = float(r2_score(actual_h, pred_h))
        mae_h = float(np.mean(np.abs(actual_h - pred_h)))
        label = f"Hour +{h + 1}"
        print(f"  {label:<12} {mape_h:>7.2f}%  {r2_h:>7.3f}  {mae_h:>10.1f}")
        horizon_metrics[label] = {"MAPE_pct": round(mape_h, 3), "R2": round(r2_h, 4), "MAE_MW": round(mae_h, 2)}

    metrics["per_horizon"] = horizon_metrics
    save_json(metrics, OUTPUTS_DIR / "lstm_metrics.json")
    print(f"\nLSTM metrics saved to {OUTPUTS_DIR / 'lstm_metrics.json'}")

    # Save predictions for comparison plots
    np.save(OUTPUTS_DIR / "lstm_predictions.npy", y_pred_mw)
    np.save(OUTPUTS_DIR / "lstm_actuals.npy", y_actual_mw)

    return metrics


if __name__ == "__main__":
    train()
