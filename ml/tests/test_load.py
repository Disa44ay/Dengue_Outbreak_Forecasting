import pandas as pd
import pytest

from ml.src.config import DATA_PATH, REQUIRED_COLUMNS, TARGET, WARD
from ml.src.load import DataValidationError, load_data, validate


@pytest.fixture(scope="module")
def data():
    return load_data(DATA_PATH)


def test_bundled_dataset_shape(data):
    assert data.shape == (31320, len(REQUIRED_COLUMNS))
    assert data[WARD].nunique() == 60
    assert data.groupby(WARD).size().eq(522).all()
    assert data["date"].min() == pd.Timestamp("2016-01-04")
    assert data["date"].max() == pd.Timestamp("2025-12-29")


def test_sorted_by_ward_then_date(data):
    assert data.groupby(WARD)["date"].apply(lambda s: s.is_monotonic_increasing).all()


def test_columns_are_case_insensitive(data):
    sample = data.head(5).rename(columns={"avg_temp_c": "avg_temp_C", "ward_id": " WARD_ID "})
    assert "avg_temp_c" in validate(sample).columns


def test_missing_column_is_clear(data):
    with pytest.raises(DataValidationError, match="rainfall_mm"):
        validate(data.head(5).drop(columns="rainfall_mm"))


def test_duplicate_ward_week_rejected(data):
    with pytest.raises(DataValidationError, match="duplicate"):
        validate(pd.concat([data.head(2), data.head(1)]))


def test_target_optional_for_inference(data):
    assert TARGET not in validate(data.head(5).drop(columns=TARGET), require_target=False).columns
