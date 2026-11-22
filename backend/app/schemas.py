from typing import Literal

from pydantic import BaseModel


class WardInfo(BaseModel):
    ward_id: str
    city: str


class DriverContribution(BaseModel):
    feature: str
    value: float
    importance: float


class PredictionResponse(BaseModel):
    ward_id: str
    horizon_weeks: Literal[2, 3, 4]
    predicted_cases: float
    risk_tier: Literal["Low", "Medium", "High", "Critical"]
    drivers: list[DriverContribution]
