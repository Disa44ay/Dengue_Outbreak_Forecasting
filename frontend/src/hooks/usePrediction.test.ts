import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { usePrediction } from "./usePrediction";
import * as api from "../lib/api";

const SAMPLE_PREDICTION = {
  ward_id: "DNCC_Ward_20",
  horizon_weeks: 2 as const,
  predicted_cases: 7.3,
  interval_low: 4.1,
  interval_high: 10.5,
  risk_tier: "High" as const,
  drivers: [],
};

describe("usePrediction", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("does not fetch and returns null prediction when wardId is null", () => {
    const spy = vi.spyOn(api, "getPrediction");

    const { result } = renderHook(() => usePrediction(null, 2));

    expect(spy).not.toHaveBeenCalled();
    expect(result.current.prediction).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  test("fetches the prediction for the given ward and horizon", async () => {
    const spy = vi.spyOn(api, "getPrediction").mockResolvedValue(SAMPLE_PREDICTION);

    const { result } = renderHook(() => usePrediction("DNCC_Ward_20", 2));

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(spy).toHaveBeenCalledWith("DNCC_Ward_20", 2);
    expect(result.current.prediction).toEqual(SAMPLE_PREDICTION);
    expect(result.current.error).toBeNull();
  });

  test("surfaces the error message when the fetch rejects", async () => {
    vi.spyOn(api, "getPrediction").mockRejectedValue(new Error("404 Not Found"));

    const { result } = renderHook(() => usePrediction("BAD_WARD", 2));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe("404 Not Found");
    expect(result.current.prediction).toBeNull();
  });

  test("refetches when the ward changes", async () => {
    const spy = vi.spyOn(api, "getPrediction").mockResolvedValue(SAMPLE_PREDICTION);

    const { result, rerender } = renderHook(({ wardId }) => usePrediction(wardId, 2), {
      initialProps: { wardId: "DNCC_Ward_20" },
    });
    await waitFor(() => expect(result.current.loading).toBe(false));

    rerender({ wardId: "NCC_Ward_03" });
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));

    expect(spy).toHaveBeenNthCalledWith(2, "NCC_Ward_03", 2);
  });
});
