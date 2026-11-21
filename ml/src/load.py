"""Load and validate dengue CSV data from the bundled file or an upload."""
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

from ml.src.config import (
    CITY,
    DATA_PATH,
    DATE,
    FEATURE_INPUT_COLUMNS,
    STATIC_COLUMNS,
    TARGET,
    WARD,
    WEATHER_COLUMNS,
)


class DataValidationError(ValueError):
    """Raised with a user-readable message when input data is invalid."""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trimmed, lowercase column names."""
    normalized = [str(column).strip().lower() for column in df.columns]
    duplicates = sorted({c for c in normalized if normalized.count(c) > 1})
    if duplicates:
        raise DataValidationError(
            f"Columns repeat after ignoring case: {', '.join(duplicates)}"
        )
    out = df.copy()
    out.columns = normalized
    return out


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Validate the schema and return clean data sorted by ward and date."""
    df = normalize_columns(df)
    required = [*FEATURE_INPUT_COLUMNS, *([TARGET] if require_target else [])]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {', '.join(missing)}")
    if df.empty:
        raise DataValidationError("The file has no data rows.")

    keep = [*FEATURE_INPUT_COLUMNS, *([TARGET] if TARGET in df.columns else [])]
    out = df[keep].copy()

    out[DATE] = pd.to_datetime(out[DATE], errors="coerce")
    numeric = [*STATIC_COLUMNS, *WEATHER_COLUMNS]
    if TARGET in out.columns:
        numeric.append(TARGET)
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    check = [*FEATURE_INPUT_COLUMNS, *([TARGET] if require_target else [])]
    null_counts = out[check].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        detail = ", ".join(f"{name} ({count})" for name, count in bad.items())
        raise DataValidationError(f"Missing or invalid values in: {detail}")

    repeated = int(out.duplicated([WARD, DATE]).sum())
    if repeated:
        raise DataValidationError(f"Found {repeated} duplicate ward-week rows.")

    if TARGET in out.columns and (out[TARGET].dropna() < 0).any():
        raise DataValidationError(f"{TARGET} cannot be negative.")

    out[CITY] = out[CITY].astype(str).str.strip()
    out[WARD] = out[WARD].astype(str).str.strip()
    return out.sort_values([WARD, DATE], kind="stable").reset_index(drop=True)


def load_data(
    source: Union[str, Path, BinaryIO] = DATA_PATH, require_target: bool = True
) -> pd.DataFrame:
    """Read a CSV path or file object and return validated data."""
    try:
        raw = pd.read_csv(source, encoding="utf-8")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"Could not read CSV: {exc}") from exc
    return validate(raw, require_target=require_target)
