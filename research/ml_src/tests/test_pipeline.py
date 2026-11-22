"""Serving pipeline tests. They use a small throwaway model, never the shipped one."""
import lightgbm as lgb
import pandas as pd
import pytest

from ml.src.config import DATE, HORIZONS, MAX_LAG, MODEL_TARGET, TARGET, WARD, WEATHER_COLUMNS
from ml.src.features import build_features, feature_columns
from ml.src.load import DataValidationError, load_data
from ml.src.pipeline import DenguePipeline
from ml.src.predict import Bundle, forecast


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def bundle(data):
    models, entries = {}, {}
    for horizon in HORIZONS:
        frame = build_features(data, horizon).dropna(subset=[MODEL_TARGET]).iloc[::10]
        models[horizon] = lgb.LGBMRegressor(objective="poisson", n_estimators=20, verbosity=-1, random_state=0).fit(
            frame[feature_columns()], frame[MODEL_TARGET]
        )
        entries[str(horizon)] = {"predicted_cuts": [1.0, 7.0, 13.0], "conformal": {"lower": -1.0, "upper": 2.0}}
    bounds = {c: {"min": float(data[c].min()), "max": float(data[c].max())} for c in WEATHER_COLUMNS}
    return Bundle(models=models, thresholds={"horizons": entries, "weather_bounds": bounds})


@pytest.fixture(scope="module")
def pipeline(data, bundle):
    return DenguePipeline.build(data, bundle)


@pytest.fixture(scope="module")
def ward(data):
    return data[WARD].iloc[0]


def next_week(pipeline, ward, weeks=1, **changes):
    last = pipeline.history[pipeline.history[WARD] == ward].iloc[-1]
    row = {column: float(last[column]) for column in WEATHER_COLUMNS}
    row[DATE] = f"{last[DATE] + pd.Timedelta(weeks=weeks):%Y-%m-%d}"
    return {**row, **changes}


def test_history_is_minimal_and_has_no_cases(pipeline, data):
    assert TARGET not in pipeline.history.columns
    assert pipeline.history.groupby(WARD).size().eq(MAX_LAG + 1).all()
    assert pipeline.history[WARD].nunique() == data[WARD].nunique()


@pytest.mark.parametrize("horizon", HORIZONS)
def test_matches_full_data_pipeline(pipeline, bundle, data, ward, horizon):
    full = build_features(data, horizon)
    row = full[(full[WARD] == ward) & (full[DATE] == full[DATE].max())]
    expected = forecast(row, horizon, bundle)["predicted"].iloc[0]
    assert pipeline.predict(ward, horizon)["predicted_cases"] == pytest.approx(expected, abs=1e-3)


def test_save_load_round_trip(pipeline, ward, tmp_path):
    path = tmp_path / "pipeline.joblib"
    pipeline.save(path)
    assert DenguePipeline.load(path).predict(ward, 3) == pipeline.predict(ward, 3)


def test_fresh_week_moves_origin_and_keeps_history(pipeline, ward):
    before = pipeline.history.copy()
    base = pipeline.predict(ward, 2)
    result = pipeline.predict(ward, 2, [next_week(pipeline, ward)])
    assert pd.Timestamp(result["origin_week"]) == pd.Timestamp(base["origin_week"]) + pd.Timedelta(weeks=1)
    assert pd.Timestamp(result["target_week"]) == pd.Timestamp(result["origin_week"]) + pd.Timedelta(weeks=2)
    pd.testing.assert_frame_equal(pipeline.history, before)


def test_replacing_a_known_week_keeps_origin(pipeline, ward):
    last = pipeline.history[pipeline.history[WARD] == ward].iloc[-1]
    row = {column: float(last[column]) for column in WEATHER_COLUMNS}
    row.update({DATE: f"{last[DATE]:%Y-%m-%d}", "rainfall_mm": float(last["rainfall_mm"]) + 50})
    assert pipeline.predict(ward, 2, [row])["origin_week"] == f"{last[DATE]:%Y-%m-%d}"


def test_out_of_range_input_warns(pipeline, ward):
    high = pipeline.bundle.thresholds["weather_bounds"]["rainfall_mm"]["max"] + 10
    assert pipeline.predict(ward, 2, [next_week(pipeline, ward, rainfall_mm=high)])["warnings"]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"rainfall_mm": -1}, "negative"),
        ({"relative_humidity_pct": 101}, "between 0 and 100"),
        ({"min_temp_c": 60}, "min <= avg <= max"),
        ({"avg_temp_c": "hot"}, "invalid"),
        ({"date": "not-a-date"}, "invalid"),
    ],
)
def test_bad_weather_rejected(pipeline, ward, changes, message):
    with pytest.raises(DataValidationError, match=message):
        pipeline.predict(ward, 2, [next_week(pipeline, ward, **changes)])


def test_gap_missing_field_ward_and_horizon_rejected(pipeline, ward):
    with pytest.raises(DataValidationError, match="consecutive"):
        pipeline.predict(ward, 2, [next_week(pipeline, ward, weeks=2)])
    partial = next_week(pipeline, ward)
    del partial["rainfall_mm"]
    with pytest.raises(DataValidationError, match="missing"):
        pipeline.predict(ward, 2, [partial])
    with pytest.raises(DataValidationError, match="Unknown ward"):
        pipeline.predict("NOPE", 2)
    with pytest.raises(DataValidationError, match="horizon"):
        pipeline.predict(ward, 7)


def test_city_ranking_is_sorted(pipeline):
    city = next(iter(pipeline.catalog()["cities"]))
    values = [item["predicted_cases"] for item in pipeline.predict_city(city, 2)]
    assert values == sorted(values, reverse=True)
