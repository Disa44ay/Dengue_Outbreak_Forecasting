import { useEffect, useState } from "react";
import { getWards } from "../lib/api";
import type { WardInfo } from "../types/api";

interface UseWardsResult {
  wards: WardInfo[];
  loading: boolean;
  error: string | null;
}

export function useWards(): UseWardsResult {
  const [wards, setWards] = useState<WardInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    getWards()
      .then((data) => {
        if (!cancelled) setWards(data);
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

  return { wards, loading, error };
}
