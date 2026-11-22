import { useMemo } from "react";
import type { WardInfo } from "../../types/api";

interface WardSelectorProps {
  wards: WardInfo[];
  selectedWardId: string | null;
  onSelect: (wardId: string) => void;
  disabled?: boolean;
}

export function WardSelector({ wards, selectedWardId, onSelect, disabled }: WardSelectorProps) {
  const byCity = useMemo(() => {
    const groups = new Map<string, WardInfo[]>();
    for (const ward of wards) {
      const list = groups.get(ward.city) ?? [];
      list.push(ward);
      groups.set(ward.city, list);
    }
    return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [wards]);

  return (
    <div className="field-group">
      <label className="field-label" htmlFor="ward-select">
        Ward
      </label>
      <div className="select-shell">
        <select
          id="ward-select"
          value={selectedWardId ?? ""}
          disabled={disabled}
          onChange={(event) => onSelect(event.target.value)}
        >
          <option value="" disabled>
            Choose a ward
          </option>
          {byCity.map(([city, cityWards]) => (
            <optgroup key={city} label={city}>
              {cityWards.map((ward) => (
                <option key={ward.ward_id} value={ward.ward_id}>
                  {ward.ward_id}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>
    </div>
  );
}
