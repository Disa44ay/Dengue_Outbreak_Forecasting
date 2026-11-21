"""Per-ward weather lags and future case targets (plan.md Task 2)."""
import pandas as pd

LAG_COLUMNS = ("avg_temp_c", "rainfall_mm", "relative_humidity_pct")
STATIC_COLUMNS = ("population_density_sqkm", "water_logging_risk_index", "sanitation_score")
LAGS = (1, 2, 3, 4)
HORIZONS = (2, 3, 4)
TARGET = "dengue_cases"


def _sorted_by_ward(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    out = _sorted_by_ward(df)
    by_ward = out.groupby("ward_id", sort=False)
    new = {f"{col}_lag{n}": by_ward[col].shift(n) for col in LAG_COLUMNS for n in lags}
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def build_lead_targets(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    out = _sorted_by_ward(df)
    by_ward = out.groupby("ward_id", sort=False)[TARGET]
    new = {f"{TARGET}_lead{h}": by_ward.shift(-h) for h in horizons}
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def feature_columns(lags: tuple[int, ...] = LAGS) -> list[str]:
    return [f"{col}_lag{n}" for col in LAG_COLUMNS for n in lags] + list(STATIC_COLUMNS)
