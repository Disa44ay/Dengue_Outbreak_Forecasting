import { MODEL_METRICS } from "../../data/modelMetrics";

export function ValidationRows() {
  return (
    <div className="board">
      <div className="board-head">Model validation, 2025 chronological test split</div>
      <div className="validation-rows">
        {MODEL_METRICS.map((row) => (
          <div className="validation-row" key={row.horizonWeeks}>
            <span className="horizon-label">{row.horizonWeeks} weeks ahead</span>
            <span className="metric">
              <span>
                <span className="metric-label">MAE</span> <span className="num">{row.mae.toFixed(3)}</span>
              </span>
              <span>
                <span className="metric-label">RMSE</span> <span className="num">{row.rmse.toFixed(3)}</span>
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
