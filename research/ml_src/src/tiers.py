"""Risk tiers from training-period case quantiles."""
import numpy as np

from ml.src.config import TIER_NAMES, TIER_QUANTILES


def compute_cuts(train_cases) -> list[float]:
    """Return the Medium, High and Critical lower bounds from training cases."""
    return [float(q) for q in np.quantile(np.asarray(train_cases, dtype=float), TIER_QUANTILES)]


def assign_tiers(values, cuts) -> np.ndarray:
    """Return tier codes 0-3. A value equal to a cut belongs to the higher tier."""
    return np.searchsorted(np.asarray(cuts, dtype=float), np.asarray(values, dtype=float), side="right")


def tier_names(codes) -> list[str]:
    return [TIER_NAMES[int(code)] for code in codes]


def rank_match_cuts(predicted, actual, actual_cuts) -> list[float]:
    """Move predicted-side cuts so predicted tier shares match actual shares.

    Call this on validation data only (docs/decisions.md, D4).
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    shares = [float((actual < cut).mean()) for cut in actual_cuts]
    cuts = [float(c) for c in np.quantile(predicted, shares)]
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = float(np.nextafter(cuts[i - 1], np.inf))
    return cuts


def tier_report(actual, predicted, actual_cuts, predicted_cuts) -> dict:
    """Score predicted tiers, focusing on High and Critical recall."""
    true_tier = assign_tiers(actual, actual_cuts)
    pred_tier = assign_tiers(predicted, predicted_cuts)
    true_high = true_tier >= 2
    pred_high = pred_tier >= 2
    hits = int((true_high & pred_high).sum())
    recall_by_tier = {}
    for code, name in enumerate(TIER_NAMES):
        members = true_tier == code
        recall_by_tier[name] = float((pred_tier[members] == code).mean()) if members.any() else None
    return {
        "high_critical_recall": hits / int(true_high.sum()) if true_high.any() else None,
        "high_critical_precision": hits / int(pred_high.sum()) if pred_high.any() else None,
        "tier_accuracy": float((true_tier == pred_tier).mean()),
        "recall_by_tier": recall_by_tier,
        "actual_high_critical_weeks": int(true_high.sum()),
        "predicted_high_critical_weeks": int(pred_high.sum()),
    }
