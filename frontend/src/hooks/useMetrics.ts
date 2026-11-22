import { useEffect, useState } from "react";
import { getMetrics } from "../lib/api";
import type { MetricsResponse } from "../types/api";

interface UseMetricsResult {
  metrics: MetricsResponse | null;
  loading: boolean;
  error: string | null;
}

export function useMetrics(): UseMetricsResult {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    getMetrics()
      .then((data) => {
        if (!cancelled) setMetrics(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { metrics, loading, error };
}
