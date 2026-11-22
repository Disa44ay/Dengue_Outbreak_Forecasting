"""Tests for ml/features.py, ml/risk.py and ml/train.py. Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from ml.features import build_lag_features, build_lead_targets, feature_columns
from ml.risk import apply_interval, compute_interval_band, compute_risk_tier, compute_thresholds
from ml.train import build_ward_snapshot, drop_incomplete_rows, load_and_split


def make_synthetic_df(wards, weeks):
    rows = []
    for w, ward in enumerate(wards):
        for i, date in enumerate(pd.date_range("2023-01-02", periods=weeks, freq="7D")):
            rows.append(
                {
                    "date": date, "city": "Dhaka", "ward_id": ward,
                    "avg_temp_c": 25.0 + w + i, "rainfall_mm": 100.0 * (w + 1) + i,
                    "relative_humidity_pct": 60.0 + w + i, "dengue_cases": 10 * (w + 1) + i,
                }
            )
    return pd.DataFrame(rows)


def test_build_lag_features_shifts_per_ward():
    out = build_lag_features(make_synthetic_df(["W1", "W2"], 6), lags=(1, 2))
    w1 = out[out.ward_id == "W1"].sort_values("date")
    w2 = out[out.ward_id == "W2"].sort_values("date")
    assert pd.isna(w1["rainfall_mm_lag1"].iloc[0])
    assert w1["rainfall_mm_lag1"].iloc[1] == w1["rainfall_mm"].iloc[0]
    assert pd.isna(w1["rainfall_mm_lag2"].iloc[1])
    assert pd.isna(w2["rainfall_mm_lag1"].iloc[0])


def test_lags_ignore_input_order():
    df = make_synthetic_df(["W1", "W2"], 6)
    shuffled = df.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(build_lag_features(df), build_lag_features(shuffled))


def test_build_lead_targets_pulls_future_value():
    out = build_lead_targets(make_synthetic_df(["W1"], 6), horizons=(2,))
    assert out["dengue_cases_lead2"].iloc[0] == out["dengue_cases"].iloc[2]
    assert pd.isna(out["dengue_cases_lead2"].iloc[-1])


def test_feature_vector_is_15_without_lag0_or_cases():
    columns = feature_columns()
    assert len(columns) == 15
    assert not any(c.endswith("_lag0") or "dengue" in c for c in columns)


def test_compute_risk_tier_boundaries():
    th = {"medium_max": 4.0, "high_max": 9.0}
    assert compute_risk_tier(0.0, th) == "Low"
    assert compute_risk_tier(4.0, th) == "Medium"
    assert compute_risk_tier(4.1, th) == "High"
    assert compute_risk_tier(9.1, th) == "Critical"


def test_low_tier_is_predictions_that_round_to_zero():
    th = {"medium_max": 4.0, "high_max": 10.0}
    assert compute_risk_tier(0.03, th) == "Low"
    assert compute_risk_tier(0.49, th) == "Low"
    assert compute_risk_tier(0.5, th) == "Medium"


def test_compute_thresholds_strictly_increasing_on_skewed_data():
    th = compute_thresholds(pd.DataFrame({"dengue_cases": [0] * 95 + [1] * 5}))
    assert th["high_max"] > th["medium_max"] > 0


def test_load_and_split_is_chronological():
    df = pd.DataFrame({"date": pd.to_datetime(["2023-06-01", "2024-06-01", "2025-06-01"])})
    train, val, test = load_and_split(df, "2023-12-31", "2024-12-31")
    assert (train["date"] <= "2023-12-31").all() and len(train) == 1
    assert ((val["date"] > "2023-12-31") & (val["date"] <= "2024-12-31")).all() and len(val) == 1
    assert (test["date"] > "2024-12-31").all() and len(test) == 1


def test_drop_incomplete_rows_removes_nan_lag_or_target():
    df = pd.DataFrame({"rainfall_mm_lag1": [1.0, None, 2.0], "dengue_cases_lead2": [3.0, 3.0, np.nan]})
    out = drop_incomplete_rows(df, ["rainfall_mm_lag1"], ["dengue_cases_lead2"])
    assert len(out) == 1


def test_build_ward_snapshot_picks_latest_row_even_if_unsorted():
    df = make_synthetic_df(["W1"], 3)
    snap = build_ward_snapshot(df.sample(frac=1, random_state=0), ["rainfall_mm"])
    assert snap["W1"]["rainfall_mm"] == pytest.approx(df.sort_values("date").iloc[-1]["rainfall_mm"])
    assert snap["W1"]["city"] == "Dhaka"


def test_compute_interval_band_widens_with_larger_residuals():
    actual = np.array([5.0, 8.0, 3.0, 10.0, 6.0, 4.0, 9.0, 2.0, 7.0, 5.0])
    predicted = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    band = compute_interval_band(actual, predicted)
    assert band["lo_q"] < 0 < band["hi_q"]


def test_apply_interval_floors_at_zero_and_scales_with_prediction():
    band = {"lo_q": -2.0, "hi_q": 2.0}
    low, high = apply_interval(0.1, band)
    assert low == 0.0  # floored, since 0.1 - 2*sqrt(0.1) would go negative
    assert high > 0.1

    low_big, high_big = apply_interval(20.0, band)
    assert (high_big - low_big) > (high - low)  # wider band for a bigger prediction
