import { useEffect, useState } from "react";
import { getPrediction } from "../lib/api";
import type { PredictionResponse } from "../types/api";

interface UsePredictionResult {
  prediction: PredictionResponse | null;
  loading: boolean;
  error: string | null;
}

export function usePrediction(
  wardId: string | null,
  horizonWeeks: 2 | 3 | 4,
): UsePredictionResult {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!wardId) {
      setPrediction(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    getPrediction(wardId, horizonWeeks)
      .then((data) => {
        if (!cancelled) setPrediction(data);
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
  }, [wardId, horizonWeeks]);

  return { prediction, loading, error };
}
