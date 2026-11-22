import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { HorizonSelector } from "./HorizonSelector";

describe("HorizonSelector", () => {
  test("marks the current value's radio as checked", () => {
    render(<HorizonSelector value={3} onChange={() => {}} />);

    expect(screen.getByRole("radio", { name: "3w" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "2w" })).not.toBeChecked();
    expect(screen.getByRole("radio", { name: "4w" })).not.toBeChecked();
  });

  test("calls onChange with the clicked horizon", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<HorizonSelector value={2} onChange={onChange} />);

    await user.click(screen.getByRole("radio", { name: "4w" }));

    expect(onChange).toHaveBeenCalledWith(4);
  });
});
