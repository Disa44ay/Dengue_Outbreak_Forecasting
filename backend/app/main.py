import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import Artifacts, load_artifacts, predict_for_ward
from .schemas import MetricsResponse, PredictionResponse, WardInfo

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts"

app = FastAPI(title="Dengue Ward Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo-only; tighten before any real deployment
    allow_methods=["GET"],
    allow_headers=["*"],
)

_artifacts: Artifacts | None = None


def _get_artifacts() -> Artifacts:
    global _artifacts
    if _artifacts is None:
        _artifacts = load_artifacts(ARTIFACTS_DIR)
    return _artifacts


@app.get("/wards", response_model=list[WardInfo])
def list_wards() -> list[WardInfo]:
    try:
        artifacts = _get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [
        WardInfo(ward_id=ward_id, city=snapshot["city"])
        for ward_id, snapshot in artifacts.ward_snapshot.items()
    ]


@app.get("/predict", response_model=PredictionResponse)
def predict(ward_id: str, horizon_weeks: int) -> PredictionResponse:
    if horizon_weeks not in (2, 3, 4):
        raise HTTPException(status_code=422, detail="horizon_weeks must be 2, 3, or 4")
    try:
        artifacts = _get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        return predict_for_ward(ward_id, horizon_weeks, artifacts)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown ward_id: {ward_id}") from exc


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="metrics.json not found — train the model first")
    return MetricsResponse(**json.loads(path.read_text()))
