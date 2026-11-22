import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { StateMessage } from "./StateMessage";

describe("StateMessage", () => {
  test("renders the title and the why line", () => {
    render(<StateMessage title="No ward selected." why="Pick a ward above." />);

    expect(screen.getByText("No ward selected.")).toBeInTheDocument();
    expect(screen.getByText("Pick a ward above.")).toBeInTheDocument();
  });

  test("renders without the why line when it is omitted", () => {
    render(<StateMessage title="No ward selected." />);

    expect(screen.getByText("No ward selected.")).toBeInTheDocument();
  });

  test("gets an alert role only when isError is true", () => {
    const { rerender } = render(<StateMessage title="Forecast failed to load." isError />);
    expect(screen.getByRole("alert")).toBeInTheDocument();

    rerender(<StateMessage title="No ward selected." />);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
