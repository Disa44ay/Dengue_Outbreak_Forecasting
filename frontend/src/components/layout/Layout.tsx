import type { ReactNode } from "react";

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="shell">
      <nav className="shell-nav">
        <span className="wordmark">Dengue Ward Risk</span>
      </nav>
      <main className="shell-main">{children}</main>
    </div>
  );
}
