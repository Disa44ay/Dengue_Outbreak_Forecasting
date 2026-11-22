const HORIZONS: Array<2 | 3 | 4> = [2, 3, 4];

interface HorizonSelectorProps {
  value: 2 | 3 | 4;
  onChange: (value: 2 | 3 | 4) => void;
}

export function HorizonSelector({ value, onChange }: HorizonSelectorProps) {
  return (
    <div className="field-group">
      <span className="field-label" id="horizon-label">
        Forecast horizon
      </span>
      <div className="choice-set" role="radiogroup" aria-labelledby="horizon-label">
        {HORIZONS.map((weeks) => (
          <label key={weeks}>
            <input
              type="radio"
              name="horizon"
              value={weeks}
              checked={value === weeks}
              onChange={() => onChange(weeks)}
            />
            <span>{weeks}w</span>
          </label>
        ))}
      </div>
    </div>
  );
}
