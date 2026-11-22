import type { PredictionResponse } from "../../types/api";

interface RiskPlateProps {
  prediction: PredictionResponse;
  city: string;
}

export function RiskPlate({ prediction, city }: RiskPlateProps) {
  const tier = prediction.risk_tier.toLowerCase();

  return (
    <div className="plate">
      <div className="plate-id">
        {prediction.ward_id} <span className="plate-city">{city}</span>
      </div>
      <div className="plate-figure">
        <span className="figure num">{prediction.predicted_cases.toFixed(1)}</span>
        <span className="unit">cases, {prediction.horizon_weeks}w out</span>
      </div>
      <span className={`tier tier--${tier}`}>
        <span className="tier-swatch" aria-hidden="true" />
        {prediction.risk_tier}
      </span>
    </div>
  );
}
