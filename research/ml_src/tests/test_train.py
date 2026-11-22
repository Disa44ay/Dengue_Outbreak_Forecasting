import numpy as np
import pandas as pd
import pytest

from ml.src.config import DATE, HORIZONS, MODEL_TARGET, PURGE_WEEKS, TARGET, TARGET_DATE, WARD
from ml.src.intervals import coverage, fit_conformal, interval_bounds
from ml.src.tiers import assign_tiers, compute_cuts, rank_match_cuts, tier_report
from ml.src.train import fit_baselines, predict_baselines, score, split_masks


@pytest.mark.parametrize("horizon", HORIZONS)
def test_splits_are_ordered_purged_and_disjoint(horizon):
    origins = pd.Series(pd.date_range("2016-01-04", "2025-12-29", freq="7D"))
    targets = origins + pd.Timedelta(weeks=horizon)
    masks = split_masks(targets)
    assert not (masks["train"] & masks["validation"]).any()
    assert not (masks["validation"] & masks["test"]).any()
    assert targets[masks["train"]].max() < origins[masks["validation"]].min()
    assert targets[masks["validation"]].max() < origins[masks["test"]].min()
    assert targets[masks["validation"]].min() == pd.Timestamp("2023-01-30")
    assert targets[masks["test"]].min() == pd.Timestamp("2024-01-29")
    assert PURGE_WEEKS >= max(HORIZONS)


def test_validation_and_test_weeks_match_across_horizons():
    origins = pd.Series(pd.date_range("2016-01-04", "2025-12-29", freq="7D"))
    for split in ("validation", "test"):
        weeks = [
            set((origins + pd.Timedelta(weeks=h))[split_masks(origins + pd.Timedelta(weeks=h))[split]])
            for h in HORIZONS
        ]
        assert weeks[0] == weeks[1] == weeks[2]


def baseline_frames():
    dates = pd.date_range("2021-01-04", periods=8, freq="7D")
    raw = pd.DataFrame({WARD: ["A"] * 8, DATE: dates, TARGET: [1, 2, 3, 4, 5, 6, 7, 8]})
    frame = raw[[WARD, DATE]].assign(**{TARGET_DATE: dates + pd.Timedelta(weeks=2)})
    frame[MODEL_TARGET] = raw[TARGET].shift(-2)
    return raw, frame.dropna().reset_index(drop=True)


def test_baselines_use_only_training_rows():
    raw, frame = baseline_frames()
    train = frame.iloc[:3]
    before = fit_baselines(train)
    changed = frame.copy()
    changed.loc[3:, MODEL_TARGET] = 999
    after = fit_baselines(changed.iloc[:3])
    pd.testing.assert_series_equal(before.ward_mean, after.ward_mean)
    pd.testing.assert_series_equal(before.ward_week_mean, after.ward_week_mean)


def test_persistence_uses_cases_at_origin():
    raw, frame = baseline_frames()
    predictions = predict_baselines(frame, fit_baselines(frame.iloc[:3]), raw)
    assert predictions["persistence"].tolist() == [1, 2, 3, 4, 5, 6]


def test_unseen_week_falls_back_to_ward_mean():
    raw, frame = baseline_frames()
    predictions = predict_baselines(frame, fit_baselines(frame.iloc[:3]), raw)
    assert predictions["ward_week_mean"][-1] == pytest.approx(predictions["ward_mean"][-1])


def test_tier_boundaries():
    assert assign_tiers([0, 0.99, 1, 6.99, 7, 12.99, 13, 40], [1, 7, 13]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_cuts_are_the_configured_quantiles():
    assert compute_cuts(np.arange(101)) == [60, 85, 95]


def test_rank_match_reproduces_actual_shares():
    rng = np.random.default_rng(1)
    actual = rng.poisson(4, 5000)
    predicted = rng.uniform(0, 3, 5000)
    cuts = rank_match_cuts(predicted, actual, [2, 5, 8])
    for actual_cut, predicted_cut in zip([2, 5, 8], cuts):
        assert (predicted < predicted_cut).mean() == pytest.approx((actual < actual_cut).mean(), abs=0.002)
    assert cuts == sorted(cuts)


def test_tier_report_recall_and_precision():
    report = tier_report([0, 8, 8, 14], [0, 8, 0, 3], [1, 7, 13], [1, 7, 13])
    assert report["high_critical_recall"] == pytest.approx(1 / 3)
    assert report["high_critical_precision"] == 1


def test_interval_is_nonnegative_and_calibrated():
    rng = np.random.default_rng(2)
    mean = rng.uniform(0.2, 10, 20000)
    calibration = rng.poisson(mean)
    holdout = rng.poisson(mean)
    conformal = fit_conformal(calibration, mean)
    lower, upper = interval_bounds(mean, conformal)
    assert (lower >= 0).all() and (upper >= lower).all()
    assert coverage(holdout, mean, conformal) == pytest.approx(0.9, abs=0.03)


def test_score_handles_zero_predictions():
    result = score([0, 3], [0, 0])
    assert result["mae"] == pytest.approx(1.5)
    assert np.isfinite(result["poisson_deviance"])
