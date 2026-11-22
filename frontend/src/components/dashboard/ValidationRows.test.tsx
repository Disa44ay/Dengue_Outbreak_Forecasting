import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { ValidationRows } from "./ValidationRows";
import { MODEL_METRICS } from "../../data/modelMetrics";

describe("ValidationRows", () => {
  test("renders one row per horizon in the published metrics", () => {
    render(<ValidationRows />);

    for (const row of MODEL_METRICS) {
      expect(screen.getByText(`${row.horizonWeeks} weeks ahead`)).toBeInTheDocument();
      expect(screen.getByText(row.mae.toFixed(3))).toBeInTheDocument();
      expect(screen.getByText(row.rmse.toFixed(3))).toBeInTheDocument();
    }
  });
});
