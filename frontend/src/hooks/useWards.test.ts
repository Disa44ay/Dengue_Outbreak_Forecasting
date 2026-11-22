import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useWards } from "./useWards";
import * as api from "../lib/api";

describe("useWards", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("starts in a loading state with no wards", () => {
    vi.spyOn(api, "getWards").mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => useWards());

    expect(result.current.loading).toBe(true);
    expect(result.current.wards).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  test("resolves with the fetched wards and clears loading", async () => {
    const wards = [{ ward_id: "DNCC_Ward_20", city: "Dhaka North" }];
    vi.spyOn(api, "getWards").mockResolvedValue(wards);

    const { result } = renderHook(() => useWards());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.wards).toEqual(wards);
    expect(result.current.error).toBeNull();
  });

  test("surfaces the error message when the fetch rejects", async () => {
    vi.spyOn(api, "getWards").mockRejectedValue(new Error("503 Service Unavailable"));

    const { result } = renderHook(() => useWards());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe("503 Service Unavailable");
    expect(result.current.wards).toEqual([]);
  });
});
