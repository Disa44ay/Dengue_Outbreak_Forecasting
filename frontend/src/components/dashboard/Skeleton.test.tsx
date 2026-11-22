import { render } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { BoardSkeleton, PlateSkeleton } from "./Skeleton";

describe("PlateSkeleton", () => {
  test("renders a single hidden placeholder block", () => {
    const { container } = render(<PlateSkeleton />);

    const el = container.querySelector(".skeleton-plate");
    expect(el).not.toBeNull();
    expect(el).toHaveAttribute("aria-hidden", "true");
  });
});

describe("BoardSkeleton", () => {
  test("renders five placeholder rows by default", () => {
    const { container } = render(<BoardSkeleton />);

    expect(container.querySelectorAll(".skeleton-row")).toHaveLength(5);
  });

  test("renders the requested number of rows", () => {
    const { container } = render(<BoardSkeleton rows={3} />);

    expect(container.querySelectorAll(".skeleton-row")).toHaveLength(3);
  });
});
