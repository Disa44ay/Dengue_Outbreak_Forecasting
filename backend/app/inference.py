import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ml.risk import compute_risk_tier

from .schemas import DriverContribution, PredictionResponse

ARTIFACT_FILES = (
    "model_h2.joblib",
    "model_h3.joblib",
    "model_h4.joblib",
    "thresholds.json",
    "feature_columns.json",
    "ward_snapshot.json",
)


@dataclass
class Artifacts:
    models: dict[int, Any]
    thresholds: dict[str, float]
    feature_columns: list[str]
    ward_snapshot: dict[str, dict]


def load_artifacts(artifacts_dir: Path) -> Artifacts:
    """Raises FileNotFoundError until ml/ (Arman, Ahnaf) has exported real artifacts."""
    missing = [f for f in ARTIFACT_FILES if not (artifacts_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"ml/artifacts/ is missing {missing} — train the model first (see plan.md Task 4)."
        )
    models = {h: joblib.load(artifacts_dir / f"model_h{h}.joblib") for h in (2, 3, 4)}
    thresholds = json.loads((artifacts_dir / "thresholds.json").read_text())
    feature_columns = json.loads((artifacts_dir / "feature_columns.json").read_text())
    ward_snapshot = json.loads((artifacts_dir / "ward_snapshot.json").read_text())
    return Artifacts(models, thresholds, feature_columns, ward_snapshot)


def predict_for_ward(ward_id: str, horizon_weeks: int, artifacts: Artifacts) -> PredictionResponse:
    snapshot = artifacts.ward_snapshot[ward_id]  # KeyError -> caller turns into 404
    vector = [snapshot[col] for col in artifacts.feature_columns]
    model = artifacts.models[horizon_weeks]
    predicted = float(model.predict([vector])[0])
    risk_tier = compute_risk_tier(predicted, artifacts.thresholds)

    # Per-prediction contributions, not feature_importances_: that's a single array fixed
    # at training time, identical for every ward/request, which would make the driver
    # breakdown show the same top-5 features no matter what was predicted. pred_contrib
    # gives each feature's actual push on *this* prediction (log-space, since the model's
    # objective is Poisson); last column is the base value, dropped here.
    contributions = model.predict([vector], pred_contrib=True)[0][:-1]
    ranked = sorted(
        zip(artifacts.feature_columns, vector, contributions),
        key=lambda t: abs(t[2]),
        reverse=True,
    )[:5]
    drivers = [
        DriverContribution(feature=f, value=float(v), importance=float(abs(c)))
        for f, v, c in ranked
    ]
    return PredictionResponse(
        ward_id=ward_id,
        horizon_weeks=horizon_weeks,
        predicted_cases=predicted,
        risk_tier=risk_tier,
        drivers=drivers,
    )
