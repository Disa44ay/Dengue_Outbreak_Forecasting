import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { WardSelector } from "./WardSelector";

const WARDS = [
  { ward_id: "DNCC_Ward_20", city: "Dhaka North" },
  { ward_id: "NCC_Ward_03", city: "Narayanganj" },
  { ward_id: "DNCC_Ward_16", city: "Dhaka North" },
];

describe("WardSelector", () => {
  test("groups ward options under their city", () => {
    render(<WardSelector wards={WARDS} selectedWardId={null} onSelect={() => {}} />);

    const dhakaGroup = screen.getByRole("group", { name: "Dhaka North" });
    expect(dhakaGroup).toBeInTheDocument();
    expect(
      Array.from(dhakaGroup.querySelectorAll("option")).map((o) => o.value),
    ).toEqual(["DNCC_Ward_20", "DNCC_Ward_16"]);
  });

  test("shows the selected ward as the current value", () => {
    render(<WardSelector wards={WARDS} selectedWardId="NCC_Ward_03" onSelect={() => {}} />);

    expect(screen.getByRole("combobox")).toHaveValue("NCC_Ward_03");
  });

  test("calls onSelect with the chosen ward id", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(<WardSelector wards={WARDS} selectedWardId={null} onSelect={onSelect} />);

    await user.selectOptions(screen.getByRole("combobox"), "NCC_Ward_03");

    expect(onSelect).toHaveBeenCalledWith("NCC_Ward_03");
  });

  test("disables the select when disabled is true", () => {
    render(<WardSelector wards={WARDS} selectedWardId={null} onSelect={() => {}} disabled />);

    expect(screen.getByRole("combobox")).toBeDisabled();
  });
});
