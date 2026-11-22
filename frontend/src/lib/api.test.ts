import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { getPrediction, getWards } from "./api";

describe("getWards", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("requests the /wards endpoint", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify([{ ward_id: "DNCC_Ward_20", city: "Dhaka North" }]), {
        status: 200,
      }),
    );

    await getWards();

    expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/\/wards$/));
  });

  test("resolves with the parsed ward list on success", async () => {
    const wards = [{ ward_id: "DNCC_Ward_20", city: "Dhaka North" }];
    vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify(wards), { status: 200 }));

    await expect(getWards()).resolves.toEqual(wards);
  });

  test("rejects with the status and body when the response is not ok", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("artifacts not found", { status: 503, statusText: "Service Unavailable" }),
    );

    await expect(getWards()).rejects.toThrow("503 Service Unavailable: artifacts not found");
  });
});

describe("getPrediction", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("requests /predict with the ward id and horizon as query params", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          ward_id: "DNCC_Ward_20",
          horizon_weeks: 3,
          predicted_cases: 7.3,
          risk_tier: "High",
          drivers: [],
        }),
        { status: 200 },
      ),
    );

    await getPrediction("DNCC_Ward_20", 3);

    const calledUrl = vi.mocked(fetch).mock.calls[0]?.[0] as string;
    expect(calledUrl).toContain("/predict?");
    expect(calledUrl).toContain("ward_id=DNCC_Ward_20");
    expect(calledUrl).toContain("horizon_weeks=3");
  });

  test("resolves with the parsed prediction on success", async () => {
    const prediction = {
      ward_id: "DNCC_Ward_20",
      horizon_weeks: 2 as const,
      predicted_cases: 7.3,
      risk_tier: "High" as const,
      drivers: [{ feature: "rainfall_mm_lag2", value: 42.1, importance: 0.81 }],
    };
    vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify(prediction), { status: 200 }));

    await expect(getPrediction("DNCC_Ward_20", 2)).resolves.toEqual(prediction);
  });

  test("rejects with the status and body when the ward is unknown", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("unknown ward_id: BAD_WARD", { status: 404, statusText: "Not Found" }),
    );

    await expect(getPrediction("BAD_WARD", 2)).rejects.toThrow(
      "404 Not Found: unknown ward_id: BAD_WARD",
    );
  });
});
