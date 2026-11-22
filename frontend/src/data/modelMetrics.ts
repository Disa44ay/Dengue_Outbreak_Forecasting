// Chronological test-split metrics from ml/artifacts/metrics.json (2025 holdout, Poisson LightGBM).
// See ARCHITECTURE.md for the model-selection rationale and the full comparison table.
export interface ModelMetric {
  horizonWeeks: 2 | 3 | 4;
  mae: number;
  rmse: number;
}

export const MODEL_METRICS: ModelMetric[] = [
  { horizonWeeks: 2, mae: 1.315061385579484, rmse: 2.4691340762326264 },
  { horizonWeeks: 3, mae: 1.492940633127397, rmse: 2.686009286636685 },
  { horizonWeeks: 4, mae: 1.6821722701305835, rmse: 3.080512543060193 },
];
