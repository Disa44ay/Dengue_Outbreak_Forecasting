import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { DriverBoard } from "./DriverBoard";
import type { DriverContribution } from "../../types/api";

const DRIVERS: DriverContribution[] = [
  { feature: "rainfall_mm_lag2", value: 42.1, importance: 0.81 },
  { feature: "relative_humidity_pct_lag1", value: 78.4, importance: 0.54 },
  { feature: "avg_temp_c_lag3", value: 29.2, importance: -0.2 },
];

describe("DriverBoard", () => {
  test("renders one row per driver with its feature name and value", () => {
    render(<DriverBoard drivers={DRIVERS} />);

    expect(screen.getByRole("row", { name: /rainfall_mm_lag2/ })).toBeInTheDocument();
    expect(screen.getByText("42.10")).toBeInTheDocument();
    expect(screen.getByText("avg_temp_c_lag3")).toBeInTheDocument();
    expect(screen.getByText("29.20")).toBeInTheDocument();
  });

  test("scales each importance bar relative to the largest absolute importance", () => {
    const { container } = render(<DriverBoard drivers={DRIVERS} />);

    const fills = container.querySelectorAll<HTMLDivElement>(".importance-bar-fill");
    expect(fills).toHaveLength(3);
    expect(fills[0].style.width).toBe("100%");
    expect(fills[2].style.width).toBe(`${(0.2 / 0.81) * 100}%`);
  });

  test("renders nothing in the body when there are no drivers", () => {
    render(<DriverBoard drivers={[]} />);

    expect(screen.queryAllByRole("row")).toHaveLength(1);
  });
});
