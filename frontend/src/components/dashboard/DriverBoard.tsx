import type { DriverContribution } from "../../types/api";

interface DriverBoardProps {
  drivers: DriverContribution[];
}

export function DriverBoard({ drivers }: DriverBoardProps) {
  const importances = drivers.map((d) => Math.abs(d.importance));
  const maxImportance = importances.length > 0 ? Math.max(...importances) || 1 : 1;

  return (
    <div className="board">
      <div className="board-head">Weather drivers behind this forecast</div>
      <table>
        <thead>
          <tr>
            <th scope="col">Feature</th>
            <th scope="col" className="num">
              Value
            </th>
            <th scope="col">Pull on prediction</th>
          </tr>
        </thead>
        <tbody>
          {drivers.map((driver) => (
            <tr key={driver.feature}>
              <td>{driver.feature}</td>
              <td className="num">{driver.value.toFixed(2)}</td>
              <td>
                <div className="importance-bar-track">
                  <div
                    className="importance-bar-fill"
                    style={{
                      width: `${(Math.abs(driver.importance) / maxImportance) * 100}%`,
                    }}
                  />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
