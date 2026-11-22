import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useMetrics } from "./useMetrics";
import * as api from "../lib/api";

describe("useMetrics", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("starts in a loading state with no metrics", () => {
    vi.spyOn(api, "getMetrics").mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => useMetrics());

    expect(result.current.loading).toBe(true);
    expect(result.current.metrics).toBeNull();
    expect(result.current.error).toBeNull();
  });

  test("resolves with the fetched metrics and clears loading", async () => {
    const metrics = {
      split_rows: { train: 24780, val: 3180, test: 2880 },
      horizons: { "2": { mae: 1.315, rmse: 2.469 } },
    };
    vi.spyOn(api, "getMetrics").mockResolvedValue(metrics);

    const { result } = renderHook(() => useMetrics());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.metrics).toEqual(metrics);
    expect(result.current.error).toBeNull();
  });

  test("surfaces the error message when the fetch rejects", async () => {
    vi.spyOn(api, "getMetrics").mockRejectedValue(new Error("503 Service Unavailable"));

    const { result } = renderHook(() => useMetrics());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe("503 Service Unavailable");
    expect(result.current.metrics).toBeNull();
  });
});
