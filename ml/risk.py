"""Risk tier thresholds and mapping (plan.md Task 3)."""
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
