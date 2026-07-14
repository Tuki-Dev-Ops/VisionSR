"use client";

import * as React from "react";
import { AnalysisPanel } from "@/components/AnalysisPanel";
import { CompareCanvas } from "@/components/CompareCanvas";
import { ControlsPanel } from "@/components/ControlsPanel";
import {
  DropOverlay,
  Dropzone,
  useClipboardPaste,
  useWindowDrop,
} from "@/components/Dropzone";
import { ErrorBanner } from "@/components/ErrorBanner";
import { JobFailure, JobProgress } from "@/components/JobProgress";
import { ResultPanel } from "@/components/ResultPanel";
import { TopBar } from "@/components/TopBar";
import { useAnalyzeSource } from "@/hooks/useAnalyzeSource";
import { useModels } from "@/hooks/useBackend";
import { useJobProgress } from "@/hooks/useJobProgress";
import { useRunJob } from "@/hooks/useRunJob";
import { resultHasAlpha } from "@/lib/models";
import { selectIsRunning, useAppStore } from "@/store/appStore";

export function Workspace() {
  const source = useAppStore((s) => s.source);
  const resultUrl = useAppStore((s) => s.job?.resultUrl ?? null);
  const result = useAppStore((s) => s.job?.result ?? null);
  const running = useAppStore(selectIsRunning);
  const { run, cancel } = useRunJob();
  const { data: models } = useModels();

  /* Whether the result is a cutout is a property of the job, not of the current
     control state: the user can flip the toggle off while the last run's cutout is
     still on screen. Read it from what actually ran. */
  const hasAlpha = result !== null && resultHasAlpha(result, models);

  useAnalyzeSource();
  useJobProgress();
  useClipboardPaste(!running);
  const dragging = useWindowDrop(!running);

  /* Ctrl/Cmd+Enter runs, Escape cancels. */
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        if (source && !running) run();
      } else if (e.key === "Escape" && running) {
        e.preventDefault();
        cancel();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [source, running, run, cancel]);

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-bg">
      <TopBar />
      <ErrorBanner />

      <main className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto p-3 lg:grid-cols-[minmax(0,1fr)_320px] lg:overflow-hidden">
        <div className="relative min-h-[360px] lg:min-h-0">
          {source ? (
            <CompareCanvas
              key={source.url}
              beforeSrc={source.url}
              afterSrc={resultUrl}
              afterHasAlpha={hasAlpha}
              afterLabel={hasAlpha ? "Cutout — RGBA" : "Enhanced"}
              overlay={
                <>
                  <JobProgress />
                  <JobFailure />
                </>
              }
            />
          ) : (
            <Dropzone dragging={dragging} />
          )}
          {dragging && source && <DropOverlay />}
        </div>

        <aside className="flex min-h-0 flex-col gap-3 lg:overflow-y-auto lg:pr-0.5">
          <ControlsPanel />
          <ResultPanel />
          <AnalysisPanel />
        </aside>
      </main>
    </div>
  );
}
