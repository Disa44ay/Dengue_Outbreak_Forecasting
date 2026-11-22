import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { RiskPlate } from "./RiskPlate";
import type { PredictionResponse } from "../../types/api";

const PREDICTION: PredictionResponse = {
  ward_id: "DNCC_Ward_20",
  horizon_weeks: 3,
  predicted_cases: 7.3,
  interval_low: 4.1,
  interval_high: 10.5,
  risk_tier: "High",
  drivers: [],
};

describe("RiskPlate", () => {
  test("shows the ward id, city, horizon and rounded predicted cases", () => {
    render(<RiskPlate prediction={PREDICTION} city="Dhaka North" />);

    expect(screen.getByText("DNCC_Ward_20")).toBeInTheDocument();
    expect(screen.getByText("Dhaka North")).toBeInTheDocument();
    expect(screen.getByText("7.3")).toBeInTheDocument();
    expect(screen.getByText(/3w out/)).toBeInTheDocument();
  });

  test("shows the risk tier label", () => {
    render(<RiskPlate prediction={PREDICTION} city="Dhaka North" />);

    expect(screen.getByText("High")).toBeInTheDocument();
  });

  test("applies a tier-specific class so each tier gets its own swatch colour", () => {
    const { container } = render(<RiskPlate prediction={PREDICTION} city="Dhaka North" />);

    expect(container.querySelector(".tier--high")).not.toBeNull();
  });

  test("shows the confidence interval", () => {
    render(<RiskPlate prediction={PREDICTION} city="Dhaka North" />);

    expect(screen.getByText(/4.1.*10.5/)).toBeInTheDocument();
  });
});
