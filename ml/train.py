"""Train the horizon models and export serving artifacts (plan.md Task 4).

Run from the repo root: python -m ml.train
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.features import HORIZONS, TARGET, build_lag_features, build_lead_targets, feature_columns
from ml.risk import compute_interval_band, compute_thresholds

ROOT = Path(__file__).resolve().parent.parent
DATA_PATHS = (ROOT / "data" / "dengue_dataset.csv", ROOT / "ml" / "data" / "dengue.csv")
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"


def load_and_split(
    df: pd.DataFrame, train_end: str = "2023-12-31", val_end: str = "2024-12-31"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date"])
    train_cut, val_cut = pd.Timestamp(train_end), pd.Timestamp(val_end)
    return df[dates <= train_cut], df[(dates > train_cut) & (dates <= val_cut)], df[dates > val_cut]


def drop_incomplete_rows(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=list(feature_cols) + list(target_cols))


def build_ward_snapshot(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    latest = df.sort_values("date").groupby("ward_id").tail(1)
    return {
        str(row["ward_id"]): {"city": str(row["city"]), **{c: float(row[c]) for c in feature_cols}}
        for _, row in latest.iterrows()
    }


def _load_data() -> pd.DataFrame:
    path = next((p for p in DATA_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Dataset not found. Expected one of: {[str(p) for p in DATA_PATHS]}")
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df


def _write_json(name: str, payload) -> None:
    with (ARTIFACTS_DIR / name).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    features = feature_columns()
    targets = [f"{TARGET}_lead{h}" for h in HORIZONS]

    full = build_lead_targets(build_lag_features(_load_data()))
    train, val, test = load_and_split(drop_incomplete_rows(full, features, targets))
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}")

    metrics = {"split_rows": {"train": len(train), "val": len(val), "test": len(test)}, "horizons": {}}
    intervals = {}
    for horizon, target in zip(HORIZONS, targets):
        model = lgb.LGBMRegressor(objective="poisson", random_state=0, verbosity=-1)
        model.fit(train[features], train[target])
        joblib.dump(model, ARTIFACTS_DIR / f"model_h{horizon}.joblib")
        predicted = model.predict(test[features])
        mae = float(mean_absolute_error(test[target], predicted))
        rmse = float(np.sqrt(mean_squared_error(test[target], predicted)))
        metrics["horizons"][str(horizon)] = {"mae": mae, "rmse": rmse}
        intervals[str(horizon)] = compute_interval_band(test[target].values, predicted)
        print(f"h={horizon} test MAE={mae:.3f} RMSE={rmse:.3f} interval_band={intervals[str(horizon)]}")

    thresholds = compute_thresholds(train)
    print(f"thresholds: {thresholds}")
    _write_json("thresholds.json", thresholds)
    _write_json("feature_columns.json", features)
    _write_json("ward_snapshot.json", build_ward_snapshot(full, features))
    _write_json("metrics.json", metrics)
    _write_json("intervals.json", intervals)


if __name__ == "__main__":
    main()
