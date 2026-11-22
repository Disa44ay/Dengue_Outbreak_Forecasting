"""Serving pipeline: one artifact that turns fresh weather input into a forecast.

    python -m ml.src.pipeline    Export ml/artifacts/pipeline.joblib (no training)

The artifact holds the trained models, thresholds and the last MAX_LAG + 1 weeks
of weather and static values per ward. It holds no case counts, and the backend
needs neither the dataset nor retraining to use it.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.config import (
    ARTIFACTS_DIR,
    CITY,
    DATA_PATH,
    DATE,
    FEATURE_INPUT_COLUMNS,
    HORIZONS,
    MAX_LAG,
    STATIC_COLUMNS,
    WARD,
    WEATHER_COLUMNS,
)
from ml.src.features import build_features
from ml.src.load import DataValidationError, load_data
from ml.src.predict import Bundle, explain, forecast, load_bundle

PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
PIPELINE_VERSION = 1


class DenguePipeline:
    """Validate weather input, build lag features and return a forecast."""

    def __init__(self, bundle: Bundle, history: pd.DataFrame):
        self.bundle = bundle
        self.history = history.sort_values([WARD, DATE], kind="stable").reset_index(drop=True)

    @classmethod
    def build(cls, data: pd.DataFrame, bundle: Bundle) -> "DenguePipeline":
        """Keep only the rows needed for lag features, without case counts."""
        ordered = data.sort_values([WARD, DATE], kind="stable")
        return cls(bundle, ordered.groupby(WARD, sort=False).tail(MAX_LAG + 1)[FEATURE_INPUT_COLUMNS])

    def save(self, path: Path = PIPELINE_PATH) -> None:
        history = self.history.assign(**{DATE: self.history[DATE].dt.strftime("%Y-%m-%d")})
        payload = {
            "version": PIPELINE_VERSION,
            "models": self.bundle.models,
            "thresholds": self.bundle.thresholds,
            "history": history.to_dict("list"),
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: Path = PIPELINE_PATH) -> "DenguePipeline":
        payload = joblib.load(path)
        if payload.get("version") != PIPELINE_VERSION:
            raise ValueError(f"Unsupported pipeline version: {payload.get('version')}")
        history = pd.DataFrame(payload["history"])
        history[DATE] = pd.to_datetime(history[DATE])
        return cls(Bundle(models=payload["models"], thresholds=payload["thresholds"]), history)

    def catalog(self) -> dict:
        """Return what a frontend needs to populate its selectors."""
        latest = self.history.groupby(WARD, sort=True).tail(1)
        cities: dict[str, list[str]] = {}
        for city, ward in zip(latest[CITY], latest[WARD]):
            cities.setdefault(city, []).append(ward)
        return {
            "cities": {city: sorted(wards) for city, wards in sorted(cities.items())},
            "horizons": list(HORIZONS),
            "latest_week": f"{self.history[DATE].max():%Y-%m-%d}",
            "weather_fields": list(WEATHER_COLUMNS),
            "weather_bounds": self.bundle.thresholds["weather_bounds"],
        }

    def _parse_weather(self, weather: list[dict]) -> tuple[pd.DataFrame, list[str]]:
        frame = pd.DataFrame(weather)
        required = [DATE, *WEATHER_COLUMNS]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise DataValidationError(f"Weather input is missing: {', '.join(missing)}")
        frame = frame[required].copy()
        frame[DATE] = pd.to_datetime(frame[DATE], errors="coerce")
        for column in WEATHER_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[WEATHER_COLUMNS].to_numpy(float)
        if frame[DATE].isna().any() or not np.isfinite(values).all():
            raise DataValidationError("Weather input has a missing or invalid value.")
        if frame[DATE].duplicated().any():
            raise DataValidationError("Weather input repeats a week.")
        if (frame["rainfall_mm"] < 0).any():
            raise DataValidationError("rainfall_mm cannot be negative.")
        if not frame["relative_humidity_pct"].between(0, 100).all():
            raise DataValidationError("relative_humidity_pct must be between 0 and 100.")
        ordered = (frame["min_temp_c"] <= frame["avg_temp_c"]) & (frame["avg_temp_c"] <= frame["max_temp_c"])
        if not ordered.all():
            raise DataValidationError("Temperatures must satisfy min <= avg <= max.")

        bounds = self.bundle.thresholds["weather_bounds"]
        warnings = [
            f"{column} is outside the training range "
            f"({bounds[column]['min']:g} to {bounds[column]['max']:g}), so the forecast is less reliable."
            for column in WEATHER_COLUMNS
            if not frame[column].between(bounds[column]["min"], bounds[column]["max"]).all()
        ]
        return frame, warnings

    def _ward_rows(self, ward_id: str, weather: list[dict] | None) -> tuple[pd.DataFrame, list[str]]:
        rows = self.history[self.history[WARD] == ward_id]
        if rows.empty:
            raise DataValidationError(f"Unknown ward: {ward_id}")
        if not weather:
            return rows.copy(), []

        fresh, warnings = self._parse_weather(weather)
        if fresh[DATE].min() < rows[DATE].min():
            raise DataValidationError(f"Weather weeks must start on or after {rows[DATE].min():%Y-%m-%d}.")
        identity = rows.iloc[-1][[CITY, WARD, *STATIC_COLUMNS]].to_dict()
        kept = rows[~rows[DATE].isin(fresh[DATE])]
        merged = pd.concat([kept, fresh.assign(**identity)[FEATURE_INPUT_COLUMNS]], ignore_index=True)
        return merged.sort_values(DATE, kind="stable").reset_index(drop=True), warnings

    def predict(self, ward_id: str, horizon: int, weather: list[dict] | None = None) -> dict:
        """Forecast one ward. Weather rows replace known weeks or extend the history."""
        if horizon not in HORIZONS:
            raise DataValidationError(f"horizon must be one of {list(HORIZONS)}.")
        rows, warnings = self._ward_rows(ward_id, weather)
        features = build_features(rows, horizon).iloc[[-1]].reset_index(drop=True)
        result = forecast(features, horizon, self.bundle).iloc[0]
        drivers = explain(features, horizon, self.bundle)
        return {
            "ward_id": result[WARD],
            "city": result[CITY],
            "horizon_weeks": int(horizon),
            "origin_week": f"{result[DATE]:%Y-%m-%d}",
            "target_week": f"{result['target_date']:%Y-%m-%d}",
            "predicted_cases": round(float(result["predicted"]), 3),
            "interval_90": [round(float(result["lower"]), 3), round(float(result["upper"]), 3)],
            "tier": result["tier"],
            "drivers": [
                {
                    "driver": item["driver"],
                    "effect_pct": round(float(item["effect_pct"]), 1),
                    "strongest_signal": item["strongest_signal"],
                    "weeks_back": None
                    if pd.isna(item["first_lag"])
                    else [int(item["first_lag"]), int(item["last_lag"])],
                }
                for item in drivers.to_dict("records")
            ],
            "warnings": warnings,
        }

    def predict_city(self, city: str, horizon: int) -> list[dict]:
        """Rank a city's wards at the latest stored week, highest risk first."""
        wards = sorted(self.history.loc[self.history[CITY] == city, WARD].unique())
        if not wards:
            raise DataValidationError(f"Unknown city: {city}")
        results = [self.predict(ward, horizon) for ward in wards]
        return sorted(results, key=lambda item: item["predicted_cases"], reverse=True)


def main() -> None:
    data = load_data(DATA_PATH)
    bundle = load_bundle()
    DenguePipeline.build(data, bundle).save()
    pipeline = DenguePipeline.load()

    worst = 0.0
    for horizon in HORIZONS:
        full = build_features(data, horizon)
        expected = forecast(full[full[DATE] == full[DATE].max()], horizon, bundle).set_index(WARD)["predicted"]
        for ward, value in expected.items():
            rows, _ = pipeline._ward_rows(ward, None)
            served = forecast(build_features(rows, horizon).iloc[[-1]], horizon, pipeline.bundle)["predicted"].iloc[0]
            worst = max(worst, abs(float(served) - float(value)))
    if worst > 1e-9:
        raise RuntimeError(f"Pipeline differs from the trained-model output by {worst}")
    size_kb = PIPELINE_PATH.stat().st_size / 1024
    print(f"Wrote {PIPELINE_PATH.name}: {size_kb:.0f} KB, {pipeline.history[WARD].nunique()} wards, "
          f"{len(pipeline.history)} history rows. Max parity difference: {worst:.1e}")


if __name__ == "__main__":
    main()
