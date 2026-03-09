"""Metrics, model comparison, and evaluation plots for electricity demand forecasting."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.utils import OUTPUTS_DIR, save_json


class ModelEvaluator:
    """Compute and compare forecasting metrics for multiple models."""

    @staticmethod
    def compute_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        model_name: str,
    ) -> dict:
        """Compute standard forecasting metrics.

        Args:
            actual: Ground-truth values (original scale).
            predicted: Model predictions (original scale).
            model_name: Label for this model.

        Returns:
            Dictionary with MAE, RMSE, MAPE, R2, max_error,
            within_5pct, within_10pct.
        """
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()

        errors = actual - predicted
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mape = float(np.mean(abs_errors / np.abs(actual)) * 100)
        r2 = float(r2_score(actual, predicted))
        max_err = float(np.max(abs_errors))
        within_5 = float(np.mean(abs_errors / np.abs(actual) < 0.05) * 100)
        within_10 = float(np.mean(abs_errors / np.abs(actual) < 0.10) * 100)

        metrics = {
            "model": model_name,
            "MAE_MW": round(mae, 3),
            "RMSE_MW": round(rmse, 3),
            "MAPE_pct": round(mape, 3),
            "R2": round(r2, 4),
            "max_error_MW": round(max_err, 3),
            "within_5pct": round(within_5, 2),
            "within_10pct": round(within_10, 2),
        }

        print(f"\n--- {model_name} Metrics ---")
        print(f"  MAE:       {mae:.2f} MW")
        print(f"  RMSE:      {rmse:.2f} MW")
        print(f"  MAPE:      {mape:.2f}%")
        print(f"  R²:        {r2:.4f}")
        print(f"  Max error: {max_err:.2f} MW")
        print(f"  Within 5%: {within_5:.1f}%")
        print(f"  Within 10%:{within_10:.1f}%")

        return metrics

    @staticmethod
    def compare_models(
        lstm_metrics: dict,
        sarima_metrics: dict,
        xgboost_metrics: dict | None = None,
    ) -> dict:
        """Print side-by-side comparison table and save to JSON.

        Supports 2-model (LSTM vs SARIMA) or 3-model (+ XGBoost) comparison.

        Args:
            lstm_metrics: Metrics dict for LSTM.
            sarima_metrics: Metrics dict for SARIMA.
            xgboost_metrics: Optional metrics dict for XGBoost.

        Returns:
            Combined comparison dictionary.
        """
        has_xgb = xgboost_metrics is not None
        col_w = 13

        header = f"{'Metric':<20} {'LSTM':>{col_w}} {'SARIMA':>{col_w}}"
        if has_xgb:
            header += f" {'XGBoost*':>{col_w}}"
        sep = "=" * (20 + col_w * (3 if has_xgb else 2) + (3 if has_xgb else 2))

        print("\n" + sep)
        print(header)
        print(sep)

        rows = [
            ("MAE (MW)", "MAE_MW"),
            ("RMSE (MW)", "RMSE_MW"),
            ("MAPE (%)", "MAPE_pct"),
            ("R²", "R2"),
            ("Within 5%", "within_5pct"),
            ("Within 10%", "within_10pct"),
        ]
        for label, key in rows:
            def fmt(d: dict) -> str:
                v = d.get(key, "N/A")
                return f"{v:.3f}" if isinstance(v, float) else str(v)

            line = f"{label:<20} {fmt(lstm_metrics):>{col_w}} {fmt(sarima_metrics):>{col_w}}"
            if has_xgb:
                line += f" {fmt(xgboost_metrics):>{col_w}}"
            print(line)

        print(sep)
        if has_xgb:
            print("* XGBoost is 1-step-ahead (t+1); LSTM/SARIMA are longer-horizon forecasts.")

        comparison = {
            "LSTM": lstm_metrics,
            "SARIMA": sarima_metrics,
        }
        if has_xgb:
            comparison["XGBoost"] = xgboost_metrics

        save_json(comparison, OUTPUTS_DIR / "model_comparison.json")
        print(f"\nComparison saved to {OUTPUTS_DIR / 'model_comparison.json'}")
        return comparison


def plot_predictions(
    actual: np.ndarray,
    lstm_pred: np.ndarray,
    sarima_pred: np.ndarray,
    xgboost_pred: np.ndarray | None = None,
    index: pd.DatetimeIndex | None = None,
) -> None:
    """Generate and save five evaluation plots.

    Plots:
        1. Full test period: actual vs all models
        2. First 2 weeks zoomed
        3. Scatter: actual vs predicted (one panel per model)
        4. Residuals over time
        5. Error distribution histogram

    Args:
        actual: Ground-truth values.
        lstm_pred: LSTM predictions (same length as actual).
        sarima_pred: SARIMA predictions (same length as actual).
        xgboost_pred: Optional XGBoost predictions (same length as actual).
        index: Optional DatetimeIndex for x-axis labels.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    actual = np.array(actual).flatten()
    lstm_pred = np.array(lstm_pred).flatten()
    sarima_pred = np.array(sarima_pred).flatten()

    # Build model list dynamically
    models = [
        ("LSTM", lstm_pred, "steelblue"),
        ("SARIMA", sarima_pred, "darkorange"),
    ]
    if xgboost_pred is not None:
        xgboost_pred = np.array(xgboost_pred).flatten()
        models.append(("XGBoost", xgboost_pred, "seagreen"))

    # Align lengths (XGBoost may be 1 shorter due to shift)
    min_len = min(len(actual), *(len(p) for _, p, _ in models))
    actual = actual[:min_len]
    models = [(name, pred[:min_len], color) for name, pred, color in models]

    x = index[:min_len] if index is not None else np.arange(min_len)
    two_weeks = min(336, min_len)

    # --- Figure 1: Full test period ---
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, actual, label="Actual", color="black", linewidth=0.8, alpha=0.9)
    for name, pred, color in models:
        ax.plot(x, pred, label=name, color=color, linewidth=0.8, alpha=0.8)
    ax.set_title("Full Test Period: Actual vs Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig1_full_test_period.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: First 2 weeks zoomed ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x[:two_weeks], actual[:two_weeks], label="Actual", color="black", linewidth=1.2)
    for name, pred, color in models:
        ax.plot(x[:two_weeks], pred[:two_weeks], label=name, color=color, linewidth=1.2)
    ax.set_title("First 2 Weeks of Test Period (Zoomed)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig2_two_week_zoom.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Scatter plots (one panel per model) ---
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, pred, color) in zip(axes, models):
        ax.scatter(actual, pred, alpha=0.3, s=5, color=color)
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")
        r2 = float(np.corrcoef(actual, pred)[0, 1] ** 2)
        ax.set_xlabel("Actual (MW)")
        ax.set_ylabel("Predicted (MW)")
        ax.set_title(f"{name}: Actual vs Predicted (R²={r2:.3f})")
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig3_scatter.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: Residuals over time ---
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    for ax, (name, pred, color) in zip(axes, models):
        residuals = actual - pred
        ax.plot(x, residuals, color=color, linewidth=0.6, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Residual (MW)")
        ax.set_title(f"{name} Residuals Over Time")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig4_residuals.png", dpi=150)
    plt.close(fig)

    # --- Figure 5: Error distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, pred, color in models:
        errors = actual - pred
        ax.hist(errors, bins=60, alpha=0.55, color=color, label=f"{name} errors")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Prediction Error (MW)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution Comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig5_error_distribution.png", dpi=150)
    plt.close(fig)

    print(f"All 5 plots saved to {OUTPUTS_DIR}/")
