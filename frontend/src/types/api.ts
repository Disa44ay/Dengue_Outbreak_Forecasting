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
  interval_low: number;
  interval_high: number;
  risk_tier: RiskTier;
  drivers: DriverContribution[];
}

export interface HorizonMetrics {
  mae: number;
  rmse: number;
}

export interface MetricsResponse {
  split_rows: Record<string, number>;
  horizons: Record<string, HorizonMetrics>;
}
