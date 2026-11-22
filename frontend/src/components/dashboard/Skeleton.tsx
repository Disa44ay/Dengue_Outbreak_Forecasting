export function PlateSkeleton() {
  return <div className="skeleton skeleton-plate" aria-hidden="true" />;
}

export function BoardSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div aria-hidden="true">
      {Array.from({ length: rows }, (_, i) => (
        <div className="skeleton skeleton-row" key={i} />
      ))}
    </div>
  );
}
