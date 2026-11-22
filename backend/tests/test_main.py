from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_wards_returns_the_60_snapshot_wards():
    resp = client.get("/wards")
    assert resp.status_code == 200
    assert len(resp.json()) == 60


def test_predict_rejects_invalid_horizon():
    resp = client.get("/predict", params={"ward_id": "anything", "horizon_weeks": 5})
    assert resp.status_code == 422


def test_predict_unknown_ward_returns_404():
    resp = client.get("/predict", params={"ward_id": "NOT_A_REAL_WARD", "horizon_weeks": 2})
    assert resp.status_code == 404


def test_driver_breakdown_varies_by_ward():
    # Regression guard: drivers must come from per-prediction pred_contrib, not a single
    # feature_importances_ array — which would be identical for every ward and request.
    wards = [w["ward_id"] for w in client.get("/wards").json()[:3]]
    top_drivers = {
        ward: client.get("/predict", params={"ward_id": ward, "horizon_weeks": 2}).json()["drivers"][0]["feature"]
        for ward in wards
    }
    assert len(set(top_drivers.values())) > 1, f"top driver never changes across wards: {top_drivers}"
