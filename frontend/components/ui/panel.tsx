"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

export function Panel({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className={cn(
        "rounded-lg border border-border bg-panel",
        className,
      )}
    >
      {children}
    </section>
  );
}

export function PanelHeader({
  title,
  actions,
  className,
}: {
  title: string;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn(
        "flex h-9 items-center justify-between gap-2 border-b border-border px-3",
        className,
      )}
    >
      <h2 className="text-[11px] font-medium tracking-wider text-fg-subtle uppercase">
        {title}
      </h2>
      {actions}
    </header>
  );
}

export function PanelBody({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  return <div className={cn("space-y-3 p-3", className)}>{children}</div>;
}

/** Label on the left, value on the right — the workhorse row of this UI. */
export function Row({
  label,
  value,
  title,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  title?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-[12px]">
      <span className="shrink-0 text-fg-subtle" title={title}>
        {label}
      </span>
      <span className="tabular min-w-0 truncate text-right text-fg">
        {value}
      </span>
    </div>
  );
}
