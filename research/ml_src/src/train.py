"""Train one Poisson LightGBM model per horizon and write evaluation artifacts.

    python -m ml.src.train                    Full run, including the 2024-25 test
    python -m ml.src.train --validation-only  Keeps the test years unscored
"""
import argparse
import json
import platform
import time
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from ml.src.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    DATE,
    EARLY_STOPPING_ROUNDS,
    HORIZONS,
    LAGS,
    LGBM_PARAMS,
    MIN_HIGH_CRITICAL_RECALL,
    MODEL_TARGET,
    PURGE_WEEKS,
    TARGET,
    TARGET_DATE,
    TEST_END,
    TEST_START,
    TIER_NAMES,
    TRAIN_END,
    VAL_END,
    VAL_START,
    WARD,
    WEATHER_COLUMNS,
)
from ml.src.features import build_features, feature_columns
from ml.src.intervals import coverage, fit_conformal
from ml.src.load import load_data
from ml.src.tiers import compute_cuts, rank_match_cuts, tier_report

DEVIANCE_FLOOR = 1e-3


@dataclass
class Baselines:
    ward_mean: pd.Series
    ward_week_mean: pd.Series
    global_mean: float


def split_masks(target_dates: pd.Series) -> dict[str, pd.Series]:
    """Split by target date. Validation and test drop their first purge weeks."""
    purge = pd.Timedelta(weeks=PURGE_WEEKS)
    return {
        "train": target_dates <= pd.Timestamp(TRAIN_END),
        "validation": (target_dates >= pd.Timestamp(VAL_START) + purge)
        & (target_dates <= pd.Timestamp(VAL_END)),
        "test": (target_dates >= pd.Timestamp(TEST_START) + purge)
        & (target_dates <= pd.Timestamp(TEST_END)),
    }


def _iso_week(dates: pd.Series) -> pd.Series:
    return dates.dt.isocalendar().week.astype(int)


def fit_baselines(train: pd.DataFrame) -> Baselines:
    """Fit mean baselines from training rows only."""
    weeks = _iso_week(train[TARGET_DATE])
    return Baselines(
        ward_mean=train.groupby(WARD)[MODEL_TARGET].mean(),
        ward_week_mean=train.groupby([train[WARD], weeks])[MODEL_TARGET].mean(),
        global_mean=float(train[MODEL_TARGET].mean()),
    )


def predict_baselines(frame: pd.DataFrame, baselines: Baselines, raw: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return ward-mean, persistence and ward x week-of-year predictions."""
    ward_mean = frame[WARD].map(baselines.ward_mean).fillna(baselines.global_mean).to_numpy(float)
    keys = pd.MultiIndex.from_arrays([frame[WARD], _iso_week(frame[TARGET_DATE])])
    ward_week = baselines.ward_week_mean.reindex(keys).to_numpy(float)
    ward_week = np.where(np.isnan(ward_week), ward_mean, ward_week)
    origin_keys = pd.MultiIndex.from_frame(frame[[WARD, DATE]])
    persistence = raw.set_index([WARD, DATE])[TARGET].reindex(origin_keys).to_numpy(float)
    return {"ward_mean": ward_mean, "persistence": persistence, "ward_week_mean": ward_week}


def score(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "poisson_deviance": float(
            mean_poisson_deviance(actual, np.clip(predicted, DEVIANCE_FLOOR, None))
        ),
    }


def score_split(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    """Score every predictor overall and per target year."""
    actual = frame[MODEL_TARGET].to_numpy(float)
    years = frame[TARGET_DATE].dt.year.to_numpy()
    by_year = {}
    for year in sorted(set(years.tolist())):
        in_year = years == year
        by_year[str(year)] = {
            name: score(actual[in_year], values[in_year]) for name, values in predictions.items()
        }
    return {
        "rows": int(len(frame)),
        "overall": {name: score(actual, values) for name, values in predictions.items()},
        "by_year": by_year,
    }


def fit_model(train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        train[columns],
        train[MODEL_TARGET],
        eval_X=validation[columns],
        eval_y=validation[MODEL_TARGET],
        eval_metric="poisson",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def add_case_lags(frame: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add per-ward case lags at the origin, for the ablation only."""
    lagged = raw[[WARD, DATE]].copy()
    names = [f"cases_lag{lag}" for lag in LAGS]
    by_ward = raw.groupby(WARD, sort=False)[TARGET]
    for lag, name in zip(LAGS, names):
        lagged[name] = by_ward.shift(lag)
    return frame.merge(lagged, on=[WARD, DATE], how="left", validate="one_to_one"), names


def case_lag_ablation(train, validation, raw, base_prediction) -> dict:
    """Compare validation error with and without case lags."""
    train_plus, names = add_case_lags(train, raw)
    validation_plus, _ = add_case_lags(validation, raw)
    columns = [*feature_columns(), *names]
    model = fit_model(train_plus, validation_plus, columns)
    actual = validation[MODEL_TARGET]
    return {
        "split": "validation",
        "weather_only": score(actual, base_prediction),
        "with_case_lags": score(actual, model.predict(validation_plus[columns])),
    }


def measure_inference(path, frame: pd.DataFrame, repeats: int = 30) -> dict:
    """Measure load and prediction time for a saved model."""
    columns = feature_columns()
    start = time.perf_counter()
    model = joblib.load(path)["model"]
    load_ms = (time.perf_counter() - start) * 1000
    latest = frame[frame[DATE] == frame[DATE].max()][columns]

    def median_ms(call) -> float:
        times = []
        for _ in range(repeats):
            begin = time.perf_counter()
            call()
            times.append((time.perf_counter() - begin) * 1000)
        return float(np.median(times))

    return {
        "model_size_kb": path.stat().st_size / 1024,
        "load_ms": load_ms,
        "predict_one_ward_ms": median_ms(lambda: model.predict(latest.iloc[[0]])),
        "predict_all_wards_ms": median_ms(lambda: model.predict(latest)),
        "drivers_all_wards_ms": median_ms(lambda: model.predict(latest, pred_contrib=True)),
        "wards_in_batch": int(len(latest)),
    }


def _rounded(value):
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def _write_json(name: str, payload: dict) -> None:
    with (ARTIFACTS_DIR / name).open("w", encoding="utf-8") as handle:
        json.dump(_rounded(payload), handle, indent=2)
        handle.write("\n")


def train_horizon(raw: pd.DataFrame, horizon: int, actual_cuts: list[float], include_test: bool) -> tuple[dict, dict]:
    """Train and evaluate one horizon. Returns metrics and threshold entries."""
    columns = feature_columns()
    frame = build_features(raw, horizon)
    masks = split_masks(frame[TARGET_DATE])
    train = frame[masks["train"]].reset_index(drop=True)
    validation = frame[masks["validation"]].reset_index(drop=True)

    model = fit_model(train, validation, columns)
    baselines = fit_baselines(train)
    val_pred = model.predict(validation[columns])
    val_actual = validation[MODEL_TARGET].to_numpy(float)

    conformal = fit_conformal(val_actual, val_pred)
    before = tier_report(val_actual, val_pred, actual_cuts, actual_cuts)
    rank_matched = before["high_critical_recall"] < MIN_HIGH_CRITICAL_RECALL
    predicted_cuts = rank_match_cuts(val_pred, val_actual, actual_cuts) if rank_matched else list(actual_cuts)

    path = ARTIFACTS_DIR / f"model_h{horizon}.joblib"
    joblib.dump({"model": model, "horizon": horizon, "features": columns}, path, compress=3)

    metrics = {
        "best_iteration": int(model.best_iteration_),
        "train_rows": int(len(train)),
        "validation": {
            **score_split(validation, {"model": val_pred, **predict_baselines(validation, baselines, raw)}),
            "tiers": tier_report(val_actual, val_pred, actual_cuts, predicted_cuts),
            "tiers_before_rank_match": before,
            "interval_coverage_90": coverage(val_actual, val_pred, conformal),
        },
        "test": None,
        "ablation_case_lags": case_lag_ablation(train, validation, raw, val_pred),
        "inference": measure_inference(path, frame),
    }
    if include_test:
        test = frame[masks["test"]].reset_index(drop=True)
        test_pred = model.predict(test[columns])
        test_actual = test[MODEL_TARGET].to_numpy(float)
        metrics["test"] = {
            **score_split(test, {"model": test_pred, **predict_baselines(test, baselines, raw)}),
            "tiers": tier_report(test_actual, test_pred, actual_cuts, predicted_cuts),
            "interval_coverage_90": coverage(test_actual, test_pred, conformal),
        }
    entry = {"predicted_cuts": predicted_cuts, "rank_matched": rank_matched, "conformal": conformal}
    return metrics, entry


def print_summary(metrics: dict, split: str) -> None:
    print(f"\n{split.upper()} MAE")
    print(f"{'h':>2} {'model':>7} {'ward':>7} {'persist':>8} {'ward-wk':>8} {'HC rec':>7} {'HC prec':>8} {'cover':>6} {'trees':>6}")
    for horizon, entry in metrics["horizons"].items():
        result = entry[split]
        o = result["overall"]
        t = result["tiers"]
        print(
            f"{horizon:>2} {o['model']['mae']:7.3f} {o['ward_mean']['mae']:7.3f} {o['persistence']['mae']:8.3f} "
            f"{o['ward_week_mean']['mae']:8.3f} {t['high_critical_recall']:7.3f} {t['high_critical_precision']:8.3f} "
            f"{result['interval_coverage_90']:6.3f} {entry['best_iteration']:6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-only", action="store_true", help="do not score the test years")
    args = parser.parse_args()
    include_test = not args.validation_only

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    raw = load_data(DATA_PATH)
    train_period = raw[raw[DATE] <= pd.Timestamp(TRAIN_END)]
    actual_cuts = compute_cuts(train_period[TARGET])

    horizons_metrics, horizons_thresholds = {}, {}
    for horizon in HORIZONS:
        metrics, entry = train_horizon(raw, horizon, actual_cuts, include_test)
        horizons_metrics[str(horizon)] = metrics
        horizons_thresholds[str(horizon)] = entry

    thresholds = {
        "tier_names": list(TIER_NAMES),
        "actual_cuts": actual_cuts,
        "horizons": horizons_thresholds,
        "weather_bounds": {
            column: {"min": float(train_period[column].min()), "max": float(train_period[column].max())}
            for column in WEATHER_COLUMNS
        },
    }
    payload = {
        "status": "final" if include_test else "validation_only",
        "split": {
            "train_target_end": TRAIN_END,
            "validation": [VAL_START, VAL_END],
            "test": [TEST_START, TEST_END],
            "purge_weeks": PURGE_WEEKS,
        },
        "environment": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "params": {k: v for k, v in LGBM_PARAMS.items() if k != "verbosity"},
        },
        "horizons": horizons_metrics,
    }
    _write_json("thresholds.json", thresholds)
    _write_json("metrics.json", payload)

    print_summary(payload, "validation")
    if include_test:
        print_summary(payload, "test")
    print(f"\nTier cuts: {actual_cuts}  rank-matched: { {h: e['rank_matched'] for h, e in horizons_thresholds.items()} }")
    for horizon, entry in horizons_metrics.items():
        ablation = entry["ablation_case_lags"]
        print(
            f"Ablation h={horizon}: validation MAE {ablation['weather_only']['mae']:.3f} weather-only, "
            f"{ablation['with_case_lags']['mae']:.3f} with case lags"
        )
    for horizon, entry in horizons_metrics.items():
        speed = entry["inference"]
        print(
            f"Speed h={horizon}: {speed['model_size_kb']:.0f} KB, load {speed['load_ms']:.0f} ms, "
            f"one ward {speed['predict_one_ward_ms']:.2f} ms, {speed['wards_in_batch']} wards "
            f"{speed['predict_all_wards_ms']:.2f} ms, drivers {speed['drivers_all_wards_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
