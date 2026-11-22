import { describe, expect, test } from "vitest";
import { parseHorizon } from "./horizon";

describe("parseHorizon", () => {
  test("returns the parsed value for a valid horizon string", () => {
    expect(parseHorizon("3")).toBe(3);
  });

  test("defaults to 2 when given null", () => {
    expect(parseHorizon(null)).toBe(2);
  });

  test("defaults to 2 when given an out-of-range number", () => {
    expect(parseHorizon("7")).toBe(2);
  });

  test("defaults to 2 when given non-numeric text", () => {
    expect(parseHorizon("abc")).toBe(2);
  });
});
