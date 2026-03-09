"""Feature engineering and train/test splitting for electricity demand forecasting."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import get_data_path


FEATURE_COLS = [
    "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
    "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
    "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav",
    "holiday", "school", "Holiday_ID",
    # engineered features added by engineer_features()
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24",
]

TARGET_COL = "nat_demand"

TRAIN_END = "2019-12-31 23:00:00"
TEST_START = "2020-01-01 00:00:00"


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset and parse the datetime index.

    Args:
        path: Optional explicit path to the CSV. Auto-detected if None.

    Returns:
        DataFrame with DatetimeIndex.
    """
    if path is None:
        path = get_data_path()

    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df = df.sort_index()
    print(f"Loaded data: {len(df):,} rows from {df.index.min()} to {df.index.max()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and lag features to the DataFrame.

    Engineers:
        - hour, day_of_week, month, is_weekend
        - Cyclical encodings: hour_sin/cos, month_sin/cos
        - Lag features: lag_24, lag_168
        - Rolling stats: rolling_mean_24, rolling_std_24

    Drops rows with NaN values introduced by lag/rolling features.

    Args:
        df: Raw DataFrame with DatetimeIndex and nat_demand column.

    Returns:
        DataFrame with all engineered features, NaN rows removed.
    """
    df = df.copy()

    # Basic temporal features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features on target
    df["lag_24"] = df[TARGET_COL].shift(24)
    df["lag_168"] = df[TARGET_COL].shift(168)

    # Rolling stats
    df["rolling_mean_24"] = df[TARGET_COL].rolling(window=24, min_periods=24).mean()
    df["rolling_std_24"] = df[TARGET_COL].rolling(window=24, min_periods=24).std()

    # Drop NaN rows (from lag/rolling)
    rows_before = len(df)
    df = df.dropna()
    print(f"Dropped {rows_before - len(df)} rows with NaN (from lag/rolling features). "
          f"Remaining: {len(df):,}")

    return df


def temporal_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame temporally into train and test sets.

    Train: 2015-01-01 to 2019-12-31
    Test:  2020-01-01 to 2020-06-27

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df = df[df.index <= TRAIN_END]
    test_df = df[df.index >= TEST_START]

    print(f"Train size: {len(train_df):,} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Test size:  {len(test_df):,} rows ({test_df.index.min()} to {test_df.index.max()})")
    print(f"Split date: {TEST_START}")

    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns present in df (excluding target)."""
    base = [
        "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
        "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
        "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav",
        "holiday", "school", "Holiday_ID",
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24",
    ]
    return [c for c in base if c in df.columns]


if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    train_df, test_df = temporal_train_test_split(df)
    print(f"\nFeature columns ({len(get_feature_columns(df))}):")
    print(get_feature_columns(df))
