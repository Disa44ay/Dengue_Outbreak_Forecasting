import type { MetricsResponse } from "../../types/api";

interface ValidationRowsProps {
  metrics: MetricsResponse | null;
  loading: boolean;
  error: string | null;
}

export function ValidationRows({ metrics, loading, error }: ValidationRowsProps) {
  const rows = metrics
    ? Object.entries(metrics.horizons)
        .map(([horizon, row]) => ({ horizonWeeks: Number(horizon), ...row }))
        .sort((a, b) => a.horizonWeeks - b.horizonWeeks)
    : [];

  return (
    <div className="board">
      <div className="board-head">Model validation, 2025 chronological test split</div>
      {loading && <div className="validation-row">Loading validation metrics&hellip;</div>}
      {error && (
        <div className="validation-row">
          Validation metrics failed to load. {error}
        </div>
      )}
      {!loading && !error && (
        <ul className="validation-rows">
          {rows.map((row) => (
            <li className="validation-row" key={row.horizonWeeks}>
              <span className="horizon-label">{row.horizonWeeks} weeks ahead</span>
              <span className="metric">
                <span>
                  <span className="metric-label">MAE</span> <span className="num">{row.mae.toFixed(3)}</span>
                </span>
                <span>
                  <span className="metric-label">RMSE</span> <span className="num">{row.rmse.toFixed(3)}</span>
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
