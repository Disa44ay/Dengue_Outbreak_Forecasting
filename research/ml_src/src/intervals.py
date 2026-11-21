"""Conformal forecast bands scaled by the Poisson standard deviation."""
import numpy as np

from ml.src.config import INTERVAL_QUANTILES

MIN_SCALE = 1e-6


def _scale(predicted) -> np.ndarray:
    return np.sqrt(np.clip(np.asarray(predicted, dtype=float), MIN_SCALE, None))


def fit_conformal(actual, predicted) -> dict:
    """Fit scaled residual quantiles on calibration (validation) data."""
    residuals = (np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)) / _scale(predicted)
    lower, upper = np.quantile(residuals, INTERVAL_QUANTILES)
    return {"lower": float(lower), "upper": float(upper)}


def interval_bounds(predicted, conformal: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds, with the lower bound floored at zero."""
    predicted = np.asarray(predicted, dtype=float)
    scale = _scale(predicted)
    lower = np.clip(predicted + conformal["lower"] * scale, 0, None)
    upper = np.clip(predicted + conformal["upper"] * scale, 0, None)
    return lower, upper


def coverage(actual, predicted, conformal: dict) -> float:
    """Return the share of actual values inside the band."""
    actual = np.asarray(actual, dtype=float)
    lower, upper = interval_bounds(predicted, conformal)
    return float(((actual >= lower) & (actual <= upper)).mean())
