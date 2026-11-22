const VALID_HORIZONS = [2, 3, 4] as const;

export function parseHorizon(raw: string | null): 2 | 3 | 4 {
  const parsed = Number(raw);
  return (VALID_HORIZONS as readonly number[]).includes(parsed) ? (parsed as 2 | 3 | 4) : 2;
}
