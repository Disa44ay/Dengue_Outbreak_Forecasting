import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { Layout } from "../components/layout/Layout";
import { WardSelector } from "../components/dashboard/WardSelector";
import { HorizonSelector } from "../components/dashboard/HorizonSelector";
import { RiskPlate } from "../components/dashboard/RiskPlate";
import { DriverBoard } from "../components/dashboard/DriverBoard";
import { ValidationRows } from "../components/dashboard/ValidationRows";
import { PlateSkeleton, BoardSkeleton } from "../components/dashboard/Skeleton";
import { StateMessage } from "../components/dashboard/StateMessage";
import { useWards } from "../hooks/useWards";
import { usePrediction } from "../hooks/usePrediction";
import { useMetrics } from "../hooks/useMetrics";
import { parseHorizon } from "../lib/horizon";

export function Dashboard() {
  const { wards, loading: wardsLoading, error: wardsError } = useWards();
  const { metrics, loading: metricsLoading, error: metricsError } = useMetrics();
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedWardId = searchParams.get("ward");
  const horizonWeeks = parseHorizon(searchParams.get("horizon"));

  const selectedWard = useMemo(
    () => wards.find((w) => w.ward_id === selectedWardId) ?? null,
    [wards, selectedWardId],
  );

  const { prediction, loading: predictionLoading, error: predictionError } = usePrediction(
    selectedWardId,
    horizonWeeks,
  );

  function setWard(wardId: string) {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("ward", wardId);
      return next;
    });
  }

  function setHorizon(weeks: 2 | 3 | 4) {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("horizon", String(weeks));
      return next;
    });
  }

  return (
    <Layout>
      <div className="controls">
        <WardSelector
          wards={wards}
          selectedWardId={selectedWardId}
          onSelect={setWard}
          disabled={wardsLoading || !!wardsError}
        />
        <HorizonSelector value={horizonWeeks} onChange={setHorizon} />
      </div>

      {wardsError && (
        <StateMessage
          title="Ward list failed to load."
          why={`${wardsError} Check that the backend is running, then reload.`}
          isError
        />
      )}

      {!wardsError && !selectedWardId && (
        <StateMessage
          title="No ward selected."
          why="Pick a ward above to see its risk forecast and the weather driving it."
        />
      )}

      {selectedWardId && predictionLoading && (
        <>
          <PlateSkeleton />
          <BoardSkeleton />
        </>
      )}

      {selectedWardId && predictionError && (
        <StateMessage
          title="Forecast failed to load."
          why={`${predictionError} Try a different ward or horizon.`}
          isError
        />
      )}

      {selectedWardId && !predictionLoading && !predictionError && prediction && selectedWard && (
        <>
          <RiskPlate prediction={prediction} city={selectedWard.city} />
          <DriverBoard drivers={prediction.drivers} />
        </>
      )}

      <ValidationRows metrics={metrics} loading={metricsLoading} error={metricsError} />
    </Layout>
  );
}
