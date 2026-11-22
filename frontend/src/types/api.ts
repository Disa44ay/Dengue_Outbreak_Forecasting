export type RiskTier = "Low" | "Medium" | "High" | "Critical";

export interface WardInfo {
  ward_id: string;
  city: string;
}

export interface DriverContribution {
  feature: string;
  value: number;
  importance: number;
}

export interface PredictionResponse {
  ward_id: string;
  horizon_weeks: 2 | 3 | 4;
  predicted_cases: number;
  risk_tier: RiskTier;
  drivers: DriverContribution[];
}
