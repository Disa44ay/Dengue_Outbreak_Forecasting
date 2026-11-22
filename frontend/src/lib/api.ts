import type { MetricsResponse, PredictionResponse, WardInfo } from "../types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export function getWards(): Promise<WardInfo[]> {
  return request<WardInfo[]>("/wards");
}

export function getPrediction(
  wardId: string,
  horizonWeeks: 2 | 3 | 4,
): Promise<PredictionResponse> {
  const params = new URLSearchParams({
    ward_id: wardId,
    horizon_weeks: String(horizonWeeks),
  });
  return request<PredictionResponse>(`/predict?${params.toString()}`);
}

export function getMetrics(): Promise<MetricsResponse> {
  return request<MetricsResponse>("/metrics");
}
