"""Local inference: forecasts, tiers, intervals, drivers and what-if scenarios."""
import json
import re
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from ml.src.config import (
    ARTIFACTS_DIR,
    CITY,
    DATE,
    HORIZONS,
    MAX_LAG,
    MODEL_TARGET,
    STATIC_COLUMNS,
    TARGET,
    TARGET_DATE,
    TEST_START,
    TRAIN_END,
    VAL_END,
    WARD,
)
from ml.src.features import build_features, feature_columns
from ml.src.intervals import interval_bounds
from ml.src.tiers import assign_tiers, tier_names

TEMPERATURE_COLUMNS = ("avg_temp_c", "max_temp_c", "min_temp_c")
FAMILIES = ("Rainfall", "Temperature", "Humidity", "Ward vulnerability", "Season")
NAMES = {
    "rainfall_mm": "rain",
    "avg_temp_c": "average temperature",
    "max_temp_c": "maximum temperature",
    "min_temp_c": "minimum temperature",
    "relative_humidity_pct": "humidity",
    "population_density_sqkm": "population density",
    "water_logging_risk_index": "water logging",
    "sanitation_score": "sanitation",
    "target_week_sin": "time of year",
    "target_week_cos": "time of year",
}


@dataclass(frozen=True)
class Bundle:
    models: dict
    thresholds: dict


def load_bundle() -> Bundle:
    """Load the serialized models and thresholds. Nothing is trained here."""
    models = {h: joblib.load(ARTIFACTS_DIR / f"model_h{h}.joblib")["model"] for h in HORIZONS}
    with (ARTIFACTS_DIR / "thresholds.json").open(encoding="utf-8") as handle:
        return Bundle(models=models, thresholds=json.load(handle))


def load_metrics() -> dict:
    with (ARTIFACTS_DIR / "metrics.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def feature_family(column: str) -> str:
    if column.startswith("rainfall_mm"):
        return "Rainfall"
    if column.startswith(TEMPERATURE_COLUMNS):
        return "Temperature"
    if column.startswith("relative_humidity_pct"):
        return "Humidity"
    if column in STATIC_COLUMNS:
        return "Ward vulnerability"
    return "Season"


def feature_weeks(column: str) -> tuple[int, int] | None:
    """Return the first and last weeks back from origin that a feature reads."""
    lagged = re.fullmatch(r".+_lag(\d+)", column)
    if lagged:
        return int(lagged.group(1)), int(lagged.group(1))
    window = re.fullmatch(r".+_(?:sum|mean)_(\d+)w", column)
    return (0, int(window.group(1)) - 1) if window else None


def describe_feature(column: str) -> str:
    """Return a plain-language name, for example 'rain 3 weeks ago'."""
    if column in NAMES:
        return NAMES[column]
    lagged = re.fullmatch(r"(.+)_lag(\d+)", column)
    if lagged:
        name, lag = NAMES[lagged.group(1)], int(lagged.group(2))
        if lag == 0:
            return f"{name} this week"
        return f"{name} {lag} week{'s' if lag > 1 else ''} ago"
    window = re.fullmatch(r"(.+)_(sum|mean)_(\d+)w", column)
    if window:
        name, kind, weeks = NAMES[window.group(1)], window.group(2), window.group(3)
        return f"{name} total, last {weeks} weeks" if kind == "sum" else f"{name} mean, last {weeks} weeks"
    return column


def _family_effects(contributions: np.ndarray, columns: list[str]) -> np.ndarray:
    """Sum log-space contributions by family. Returns rows x families."""
    families = np.array([feature_family(column) for column in columns])
    return np.column_stack([contributions[:, families == name].sum(axis=1) for name in FAMILIES])


def forecast(features: pd.DataFrame, horizon: int, bundle: Bundle, with_drivers: bool = False) -> pd.DataFrame:
    """Return predicted cases, 90% band, tier and optionally the top driver."""
    columns = feature_columns()
    X = features[columns]
    model = bundle.models[horizon]
    predicted = model.predict(X)
    entry = bundle.thresholds["horizons"][str(horizon)]
    lower, upper = interval_bounds(predicted, entry["conformal"])

    out = features[[DATE, CITY, WARD, TARGET_DATE]].copy()
    out["predicted"] = predicted
    out["lower"] = lower
    out["upper"] = upper
    out["tier"] = tier_names(assign_tiers(predicted, entry["predicted_cuts"]))
    out["actual"] = features[MODEL_TARGET].to_numpy(float) if MODEL_TARGET in features else np.nan

    if with_drivers:
        contributions = model.predict(X, pred_contrib=True)[:, :-1]
        effects = _family_effects(contributions, columns)
        strongest = np.abs(effects).argmax(axis=1)
        chosen = effects[np.arange(len(effects)), strongest]
        out["top_driver"] = [
            f"{FAMILIES[family]} ({(np.exp(value) - 1) * 100:+.0f}%)"
            for family, value in zip(strongest, chosen)
        ]
    return out


def explain(row: pd.DataFrame, horizon: int, bundle: Bundle) -> pd.DataFrame:
    """Return each driver family's percent effect on one prediction."""
    if len(row) != 1:
        raise ValueError("explain expects exactly one row")
    columns = feature_columns()
    contributions = bundle.models[horizon].predict(row[columns], pred_contrib=True)[0, :-1]
    families = np.array([feature_family(column) for column in columns])

    records = []
    for family in FAMILIES:
        indices = np.flatnonzero(families == family)
        total = float(contributions[indices].sum())
        lead = columns[indices[np.abs(contributions[indices]).argmax()]]
        weeks = feature_weeks(lead)
        records.append(
            {
                "driver": family,
                "effect_pct": (np.exp(total) - 1) * 100,
                "strongest_signal": describe_feature(lead),
                "first_lag": weeks[0] if weeks else np.nan,
                "last_lag": weeks[1] if weeks else np.nan,
            }
        )
    frame = pd.DataFrame(records)
    order = frame["effect_pct"].abs().sort_values(ascending=False).index
    return frame.loc[order].reset_index(drop=True)


def period_label(target_date: pd.Timestamp, has_actual: bool) -> str:
    """Say which evaluation period a forecast's target week belongs to."""
    if not has_actual:
        return "Live forecast (no observed cases yet)"
    if target_date <= pd.Timestamp(TRAIN_END):
        return "Training period (in-sample)"
    if target_date <= pd.Timestamp(VAL_END):
        return "Validation period (2023)"
    if target_date >= pd.Timestamp(TEST_START):
        return "Held-out test period"
    return "Outside the evaluation periods"


def apply_scenario(
    ward_rows: pd.DataFrame,
    origin: pd.Timestamp,
    weeks: int,
    bounds: dict,
    rainfall_mm: float | None = None,
    temp_shift: float = 0.0,
    humidity_shift: float = 0.0,
) -> pd.DataFrame:
    """Edit raw weather for the last N weeks up to origin, clamped to training ranges."""
    if not 1 <= weeks <= 4:
        raise ValueError("weeks must be between 1 and 4")
    out = ward_rows.copy()
    start = origin - pd.Timedelta(weeks=weeks - 1)
    mask = (out[DATE] >= start) & (out[DATE] <= origin)

    def clamped(column: str, values) -> np.ndarray:
        low, high = bounds[column]["min"], bounds[column]["max"]
        if column == "relative_humidity_pct":
            low, high = max(low, 0.0), min(high, 100.0)
        return np.clip(values, low, high)

    if rainfall_mm is not None:
        out.loc[mask, "rainfall_mm"] = clamped("rainfall_mm", np.full(int(mask.sum()), float(rainfall_mm)))
    if temp_shift:
        for column in TEMPERATURE_COLUMNS:
            out.loc[mask, column] = clamped(column, out.loc[mask, column].to_numpy(float) + temp_shift)
    if humidity_shift:
        column = "relative_humidity_pct"
        out.loc[mask, column] = clamped(column, out.loc[mask, column].to_numpy(float) + humidity_shift)
    return out


def scenario_features(ward_rows: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Rebuild the origin row with the same pipeline, using weeks <= origin only."""
    history = ward_rows[ward_rows[DATE] <= origin].drop(columns=[TARGET], errors="ignore")
    return build_features(history.tail(MAX_LAG + 1), horizon)
