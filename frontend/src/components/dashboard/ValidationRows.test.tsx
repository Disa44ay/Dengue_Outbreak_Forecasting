import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { ValidationRows } from "./ValidationRows";
import type { MetricsResponse } from "../../types/api";

const METRICS: MetricsResponse = {
  split_rows: { train: 24780, val: 3180, test: 2880 },
  horizons: {
    "2": { mae: 1.315, rmse: 2.469 },
    "3": { mae: 1.493, rmse: 2.686 },
    "4": { mae: 1.682, rmse: 3.081 },
  },
};

describe("ValidationRows", () => {
  test("renders a loading message while metrics load", () => {
    render(<ValidationRows metrics={null} loading error={null} />);

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test("renders an error message when metrics fail to load", () => {
    render(<ValidationRows metrics={null} loading={false} error="503 Service Unavailable" />);

    expect(screen.getByText(/validation metrics failed to load/i)).toBeInTheDocument();
  });

  test("renders one row per horizon, sorted ascending, with MAE and RMSE", () => {
    render(<ValidationRows metrics={METRICS} loading={false} error={null} />);

    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(3);
    expect(screen.getByText("2 weeks ahead")).toBeInTheDocument();
    expect(screen.getByText("1.315")).toBeInTheDocument();
    expect(screen.getByText("2.469")).toBeInTheDocument();
    expect(screen.getByText("4 weeks ahead")).toBeInTheDocument();
  });
});
