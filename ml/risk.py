"""Risk tier thresholds and mapping (plan.md Task 3)."""
import numpy as np
import pandas as pd


def compute_thresholds(train_df: pd.DataFrame, case_col: str = "dengue_cases") -> dict[str, float]:
    medium_max = float(train_df[case_col].quantile(0.75))
    high_max = float(train_df[case_col].quantile(0.90))
    if medium_max <= 0:
        medium_max = 1.0
    if high_max <= medium_max:
        high_max = medium_max + 1.0
    return {"medium_max": medium_max, "high_max": high_max}


def compute_risk_tier(predicted_cases: float, thresholds: dict[str, float]) -> str:
    # Poisson predictions are continuous and always above 0, but dengue_cases is an integer.
    # Below 0.5 is exactly the range that rounds to 0 cases, which is what Low means.
    if predicted_cases < 0.5:
        return "Low"
    if predicted_cases <= thresholds["medium_max"]:
        return "Medium"
    if predicted_cases <= thresholds["high_max"]:
        return "High"
    return "Critical"


def compute_interval_band(actual, predicted) -> dict[str, float]:
    # Residual-quantile band, scaled by sqrt(predicted): Poisson variance grows with the
    # mean, so a fixed-width band would be too tight for high predictions and too wide
    # for near-zero ones. 5th/95th percentile of the scaled residual gives a 90% band.
    scaled_resid = (np.asarray(actual) - np.asarray(predicted)) / np.sqrt(np.maximum(predicted, 0.01))
    lo_q, hi_q = np.percentile(scaled_resid, [5, 95])
    return {"lo_q": float(lo_q), "hi_q": float(hi_q)}


def apply_interval(predicted: float, band: dict[str, float]) -> tuple[float, float]:
    spread = max(predicted, 0.0) ** 0.5
    low = max(0.0, predicted + band["lo_q"] * spread)
    high = predicted + band["hi_q"] * spread
    return low, high
