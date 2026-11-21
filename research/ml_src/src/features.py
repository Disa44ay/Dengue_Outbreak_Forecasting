"""Build per-ward, multi-week lag features for direct h-week-ahead forecasting.

Each row is a forecast origin week t. Features use data from weeks <= t only.
The target, when case counts are available, is the ward's cases at week t+h.
"""
import numpy as np
import pandas as pd

from ml.src.config import (
    CITY,
    DATE,
    HORIZONS,
    LAGS,
    MAX_LAG,
    MEAN_COLUMNS,
    MEAN_WINDOWS,
    MODEL_TARGET,
    RAIN_SUM_WINDOWS,
    STATIC_COLUMNS,
    TARGET,
    TARGET_DATE,
    WARD,
    WEATHER_COLUMNS,
    WEEKS_PER_YEAR,
)
from ml.src.load import DataValidationError

RAIN = "rainfall_mm"
ONE_WEEK = pd.Timedelta(weeks=1)


def feature_columns() -> list[str]:
    """Return model feature names in their fixed order."""
    lags = [f"{column}_lag{lag}" for column in WEATHER_COLUMNS for lag in LAGS]
    rain_sums = [f"{RAIN}_sum_{window}w" for window in RAIN_SUM_WINDOWS]
    means = [f"{column}_mean_{window}w" for column in MEAN_COLUMNS for window in MEAN_WINDOWS]
    return [*lags, *rain_sums, *means, *STATIC_COLUMNS, "target_week_sin", "target_week_cos"]


def _check_weekly(data: pd.DataFrame) -> None:
    """Reject gaps, because row shifts are only valid on contiguous weeks."""
    steps = data.groupby(WARD, sort=False)[DATE].diff().dropna()
    irregular = steps[steps != ONE_WEEK]
    if not irregular.empty:
        wards = data.loc[irregular.index, WARD].nunique()
        raise DataValidationError(
            f"Dates must be consecutive weeks within each ward. "
            f"Found {len(irregular)} gaps across {wards} wards."
        )


def build_features(
    df: pd.DataFrame, horizon: int, drop_incomplete: bool = True
) -> pd.DataFrame:
    """Return identifiers, features and, if cases exist, the t+horizon target.

    Rows without full lag history are dropped unless drop_incomplete is False.
    Rows whose target lies beyond the data are kept with a NaN target so they
    can still be used for live forecasts.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")

    data = df.sort_values([WARD, DATE], kind="stable").reset_index(drop=True)
    _check_weekly(data)
    by_ward = data.groupby(WARD, sort=False)

    features: dict[str, pd.Series] = {}
    for column in WEATHER_COLUMNS:
        for lag in LAGS:
            features[f"{column}_lag{lag}"] = by_ward[column].shift(lag)
    for window in RAIN_SUM_WINDOWS:
        features[f"{RAIN}_sum_{window}w"] = by_ward[RAIN].transform(
            lambda s: s.rolling(window, min_periods=window).sum()
        )
    for column in MEAN_COLUMNS:
        for window in MEAN_WINDOWS:
            features[f"{column}_mean_{window}w"] = by_ward[column].transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            )
    for column in STATIC_COLUMNS:
        features[column] = data[column]

    # The target week is a known calendar value at the origin, so this is not leakage.
    target_date = data[DATE] + horizon * ONE_WEEK
    angle = 2 * np.pi * target_date.dt.isocalendar().week.astype(int).to_numpy() / WEEKS_PER_YEAR
    features["target_week_sin"] = pd.Series(np.sin(angle), index=data.index)
    features["target_week_cos"] = pd.Series(np.cos(angle), index=data.index)

    identifiers = data[[DATE, CITY, WARD]].assign(**{TARGET_DATE: target_date})
    out = pd.concat([identifiers, pd.DataFrame(features)[feature_columns()]], axis=1)
    if TARGET in data.columns:
        out[MODEL_TARGET] = by_ward[TARGET].shift(-horizon)

    if drop_incomplete:
        out = out.dropna(subset=feature_columns()).reset_index(drop=True)
        if out.empty:
            raise DataValidationError(
                f"Each ward needs at least {MAX_LAG + 1} consecutive weeks of data."
            )
    return out
