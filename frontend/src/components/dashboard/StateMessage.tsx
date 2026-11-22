interface StateMessageProps {
  title: string;
  why?: string;
  isError?: boolean;
}

export function StateMessage({ title, why, isError }: StateMessageProps) {
  return (
    <div className={`state-message${isError ? " is-error" : ""}`} role={isError ? "alert" : undefined}>
      <span className="state-title">{title}</span>
      {why && <span className="state-why">{why}</span>}
    </div>
  );
}
