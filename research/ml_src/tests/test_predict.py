"""Inference tests against the shipped models."""
import numpy as np
import pandas as pd
import pytest

from ml.src.config import DATE, HORIZONS, WARD
from ml.src.features import build_features, feature_columns
from ml.src.load import load_data
from ml.src.predict import (
    apply_scenario,
    describe_feature,
    explain,
    forecast,
    load_bundle,
    scenario_features,
)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def ward_rows():
    data = load_data()
    return data[data[WARD] == data[WARD].iloc[0]].reset_index(drop=True)


@pytest.fixture(scope="module")
def origin(ward_rows):
    return ward_rows[DATE].iloc[300]


def origin_row(rows, origin, horizon):
    frame = build_features(rows, horizon)
    return frame[frame[DATE] == origin].reset_index(drop=True)


def test_describe_feature():
    assert describe_feature("rainfall_mm_lag3") == "rain 3 weeks ago"
    assert describe_feature("rainfall_mm_lag1") == "rain 1 week ago"
    assert describe_feature("rainfall_mm_lag0") == "rain this week"
    assert describe_feature("rainfall_mm_sum_4w") == "rain total, last 4 weeks"
    assert describe_feature("water_logging_risk_index") == "water logging"


@pytest.mark.parametrize("horizon", HORIZONS)
def test_contributions_reconstruct_prediction(bundle, ward_rows, origin, horizon):
    row = origin_row(ward_rows, origin, horizon)
    model = bundle.models[horizon]
    contributions = model.predict(row[feature_columns()], pred_contrib=True)
    assert np.exp(contributions.sum(axis=1)) == pytest.approx(model.predict(row[feature_columns()]), rel=1e-6)
    assert len(explain(row, horizon, bundle)) == 5


@pytest.mark.parametrize("horizon", HORIZONS)
def test_forecast_ignores_rows_after_origin(bundle, ward_rows, origin, horizon):
    changed = ward_rows.copy()
    changed.loc[changed[DATE] > origin, ["rainfall_mm", "avg_temp_c", "dengue_cases"]] = 999
    before = forecast(origin_row(ward_rows, origin, horizon), horizon, bundle)["predicted"]
    after = forecast(origin_row(changed, origin, horizon), horizon, bundle)["predicted"]
    assert after.to_numpy() == pytest.approx(before.to_numpy())


@pytest.mark.parametrize("horizon", HORIZONS)
def test_empty_scenario_matches_baseline(bundle, ward_rows, origin, horizon):
    bounds = bundle.thresholds["weather_bounds"]
    edited = apply_scenario(ward_rows, origin, 4, bounds)
    scenario = forecast(scenario_features(edited, origin, horizon), horizon, bundle)["predicted"]
    baseline = forecast(origin_row(ward_rows, origin, horizon), horizon, bundle)["predicted"]
    assert scenario.to_numpy() == pytest.approx(baseline.to_numpy())


def test_scenario_edits_only_selected_weeks_and_clamps(bundle, ward_rows, origin):
    bounds = bundle.thresholds["weather_bounds"]
    edited = apply_scenario(ward_rows, origin, 2, bounds, rainfall_mm=1e9, humidity_shift=1e9)
    changed = edited["rainfall_mm"].ne(ward_rows["rainfall_mm"])
    allowed = ward_rows[DATE].between(origin - pd.Timedelta(weeks=1), origin)
    assert not (changed & ~allowed).any()
    assert edited.loc[allowed, "rainfall_mm"].eq(bounds["rainfall_mm"]["max"]).all()
    assert edited["relative_humidity_pct"].max() <= 100
