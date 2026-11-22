import numpy as np
import pandas as pd
import pytest

from ml.src.config import (
    CITY,
    DATA_PATH,
    DATE,
    HORIZONS,
    MAX_LAG,
    MODEL_TARGET,
    STATIC_COLUMNS,
    TARGET,
    TARGET_DATE,
    WARD,
    WEATHER_COLUMNS,
)
from ml.src.features import build_features, feature_columns
from ml.src.load import DataValidationError, load_data

WEEKS = 40
WARDS = ("W1", "W2", "W3")


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-06", periods=WEEKS, freq="7D")
    parts = []
    for ward in WARDS:
        part = pd.DataFrame({DATE: dates, CITY: "Test City", WARD: ward})
        for column in [*STATIC_COLUMNS, *WEATHER_COLUMNS]:
            part[column] = rng.uniform(1, 100, WEEKS)
        part[TARGET] = rng.poisson(3, WEEKS)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def row(features, ward, date):
    return features[(features[WARD] == ward) & (features[DATE] == date)].iloc[0]


@pytest.mark.parametrize("horizon", HORIZONS)
def test_lags_match_the_wards_own_history(frame, horizon):
    features = build_features(frame, horizon)
    origin = frame[DATE].sort_values().unique()[20]
    for ward in WARDS:
        history = frame[frame[WARD] == ward].set_index(DATE)
        current = row(features, ward, origin)
        for column in WEATHER_COLUMNS:
            for lag in range(MAX_LAG + 1):
                source = pd.Timestamp(origin) - pd.Timedelta(weeks=lag)
                assert current[f"{column}_lag{lag}"] == history.loc[source, column]


def test_rolling_rain_sum_ends_at_origin(frame):
    features = build_features(frame, 2)
    origin = pd.Timestamp(frame[DATE].sort_values().unique()[20])
    rain = frame[frame[WARD] == "W2"].set_index(DATE)["rainfall_mm"]
    expected = rain.loc[origin - pd.Timedelta(weeks=3) : origin].sum()
    assert row(features, "W2", origin)["rainfall_mm_sum_4w"] == pytest.approx(expected)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_target_is_cases_at_t_plus_h(frame, horizon):
    features = build_features(frame, horizon).dropna(subset=[MODEL_TARGET])
    actual = frame.set_index([WARD, DATE])[TARGET]
    keys = pd.MultiIndex.from_frame(features[[WARD, TARGET_DATE]])
    assert (features[TARGET_DATE] - features[DATE]).eq(pd.Timedelta(weeks=horizon)).all()
    assert features[MODEL_TARGET].to_numpy() == pytest.approx(actual.loc[keys].to_numpy())


@pytest.mark.parametrize("horizon", HORIZONS)
def test_changing_one_ward_never_changes_another(frame, horizon):
    before = build_features(frame, horizon)
    changed = frame.copy()
    in_w2 = changed[WARD] == "W2"
    changed.loc[in_w2, WEATHER_COLUMNS] = changed.loc[in_w2, WEATHER_COLUMNS] + 1000
    after = build_features(changed, horizon)
    others = before[WARD] != "W2"
    pd.testing.assert_frame_equal(before[others], after[others])
    assert not before.loc[~others, feature_columns()].equals(after.loc[~others, feature_columns()])


@pytest.mark.parametrize("horizon", HORIZONS)
def test_future_data_cannot_change_features(frame, horizon):
    cutoff = frame[DATE].sort_values().unique()[25]
    before = build_features(frame, horizon)
    changed = frame.copy()
    future = changed[DATE] > cutoff
    changed.loc[future, WEATHER_COLUMNS] = changed.loc[future, WEATHER_COLUMNS] * 50
    changed.loc[future, TARGET] = 999
    after = build_features(changed, horizon)
    known = before[DATE] <= cutoff
    pd.testing.assert_frame_equal(
        before.loc[known, feature_columns()], after.loc[known, feature_columns()]
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_truncating_future_rows_gives_identical_features(frame, horizon):
    cutoff = frame[DATE].sort_values().unique()[25]
    full = build_features(frame, horizon)
    truncated = build_features(frame[frame[DATE] <= cutoff], horizon)
    pd.testing.assert_frame_equal(
        full.loc[full[DATE] <= cutoff, feature_columns()].reset_index(drop=True),
        truncated[feature_columns()].reset_index(drop=True),
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_case_counts_never_affect_features(frame, horizon):
    changed = frame.copy()
    changed[TARGET] = 0
    pd.testing.assert_frame_equal(
        build_features(frame, horizon)[feature_columns()],
        build_features(changed, horizon)[feature_columns()],
    )
    assert not any("case" in name or name == MODEL_TARGET for name in feature_columns())


def test_row_order_does_not_matter(frame):
    shuffled = frame.sample(frac=1, random_state=3)
    pd.testing.assert_frame_equal(build_features(frame, 3), build_features(shuffled, 3))


def test_first_weeks_without_full_history_are_dropped(frame):
    features = build_features(frame, 2)
    assert features.groupby(WARD).size().eq(WEEKS - MAX_LAG).all()
    assert not features[feature_columns()].isna().any().any()


def test_latest_weeks_stay_available_for_live_forecast(frame):
    features = build_features(frame, 4)
    assert features.groupby(WARD)[MODEL_TARGET].apply(lambda s: s.isna().sum()).eq(4).all()


def test_works_without_case_column(frame):
    features = build_features(frame.drop(columns=TARGET), 2)
    assert MODEL_TARGET not in features.columns


def test_week_gap_is_rejected(frame):
    missing = frame.drop(frame[(frame[WARD] == "W1")].index[10])
    with pytest.raises(DataValidationError, match="consecutive"):
        build_features(missing, 2)


def test_too_little_history_is_rejected(frame):
    with pytest.raises(DataValidationError, match="consecutive weeks"):
        build_features(frame[frame[DATE] < frame[DATE].sort_values().unique()[MAX_LAG]], 2)


def test_invalid_horizon_is_rejected(frame):
    with pytest.raises(ValueError, match="horizon"):
        build_features(frame, 5)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_real_dataset_shape(horizon):
    features = build_features(load_data(DATA_PATH), horizon)
    assert len(features) == 60 * (522 - MAX_LAG)
    assert features[MODEL_TARGET].isna().sum() == 60 * horizon
    assert not features[feature_columns()].isna().any().any()
