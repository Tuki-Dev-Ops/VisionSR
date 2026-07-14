"use client";

import * as React from "react";
import { AlertTriangle, Radio, RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { useRunJob } from "@/hooks/useRunJob";
import { formatDuration, humanizeStage } from "@/lib/utils";
import { selectIsRunning, useAppStore } from "@/store/appStore";

function useElapsed(since: number | null): number {
  const [now, setNow] = React.useState(() => Date.now());
  React.useEffect(() => {
    if (since === null) return;
    const id = setInterval(() => setNow(Date.now()), 200);
    return () => clearInterval(id);
  }, [since]);
  return since === null ? 0 : Math.max(0, now - since);
}

/** Centre-of-canvas progress card. Rendered as an overlay on the compare view. */
export function JobProgress() {
  const job = useAppStore((s) => s.job);
  const running = useAppStore(selectIsRunning);
  const { cancel } = useRunJob();
  const elapsed = useElapsed(running && job ? job.startedAt : null);

  if (!job || !running) return null;

  const percent = Math.round(job.progress * 100);
  const indeterminate = job.progress <= 0;

  return (
    // stopPropagation: this overlay lives inside the compare canvas, whose
    // pointer handlers would otherwise capture the pointer and eat the clicks.
    <div
      className="absolute inset-0 z-20 flex items-center justify-center bg-black/45 backdrop-blur-[2px]"
      onPointerDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <div className="w-[300px] rounded-lg border border-border bg-panel p-4 shadow-2xl animate-fade-in">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate text-[13px] font-medium text-fg">
            {job.status === "queued"
              ? "Queued"
              : humanizeStage(job.stage)}
          </span>
          <span className="tabular text-[13px] text-fg-muted">
            {indeterminate ? "--" : `${percent}%`}
          </span>
        </div>

        <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-panel-inset ring-1 ring-border ring-inset">
          {indeterminate ? (
            <div className="animate-indeterminate h-full w-1/4 rounded-full bg-accent" />
          ) : (
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-200"
              style={{ width: `${Math.max(percent, 2)}%` }}
              role="progressbar"
              aria-valuenow={percent}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          )}
        </div>

        <div className="mt-3 flex items-center justify-between gap-2 text-[11px] text-fg-subtle">
          <span className="tabular">{formatDuration(elapsed)}</span>
          <div className="flex items-center gap-2">
            <Tooltip
              label={
                job.transport === "sse"
                  ? "Live updates over the event stream."
                  : "The event stream was unavailable; polling every 500 ms instead."
              }
            >
              <span className="inline-flex cursor-help items-center gap-1">
                {job.transport === "sse" ? (
                  <Radio className="size-3 text-success" />
                ) : (
                  <RefreshCw className="size-3 text-warn" />
                )}
                {job.transport === "sse" ? "live" : "polling"}
              </span>
            </Tooltip>
            <Button variant="ghost" size="sm" onClick={cancel}>
              <X /> Cancel
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Failure state, shown in place of the progress card. */
export function JobFailure() {
  const job = useAppStore((s) => s.job);
  const clearJob = useAppStore((s) => s.clearJob);
  const { run } = useRunJob();

  if (!job || job.status !== "failed") return null;

  return (
    <div
      className="absolute inset-0 z-20 flex items-center justify-center bg-black/45 p-6 backdrop-blur-[2px]"
      onPointerDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <div className="w-[360px] rounded-lg border border-danger/40 bg-panel p-4 shadow-2xl animate-fade-in">
        <div className="flex items-center gap-2 text-[13px] font-medium text-danger">
          <AlertTriangle className="size-4" />
          The job failed
        </div>
        <p className="mt-2 max-h-32 overflow-y-auto text-[12px] leading-relaxed break-words text-fg-muted">
          {job.error ?? "The backend did not say why."}
        </p>
        <div className="mt-3 flex justify-end gap-2">
          <Button variant="secondary" size="sm" onClick={clearJob}>
            Dismiss
          </Button>
          <Button variant="primary" size="sm" onClick={run}>
            <RefreshCw /> Try again
          </Button>
        </div>
      </div>
    </div>
  );
}
