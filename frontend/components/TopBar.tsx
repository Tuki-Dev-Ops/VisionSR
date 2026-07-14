"use client";

import { Cpu, FolderOpen, Moon, Sun, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { useFilePicker } from "@/components/Dropzone";
import { useHealth } from "@/hooks/useBackend";
import { useTheme } from "@/hooks/useTheme";
import { formatBytes } from "@/lib/utils";
import { selectIsRunning, useAppStore } from "@/store/appStore";

function BackendStatus() {
  const { data, isError, isLoading, refetch, isFetching } = useHealth();

  if (isLoading) {
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-fg-subtle">
        <span className="size-1.5 rounded-full bg-fg-subtle" />
        Connecting
      </span>
    );
  }

  if (isError || !data) {
    return (
      <button
        type="button"
        onClick={() => void refetch()}
        className="flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-[11px] text-danger transition-colors hover:bg-danger-soft"
      >
        <span className="size-1.5 rounded-full bg-danger" />
        {isFetching ? "Reconnecting" : "Backend offline"}
      </button>
    );
  }

  return (
    <Tooltip
      label={
        <span className="block space-y-0.5">
          <span className="block">VisionSR {data.version}</span>
          <span className="block">Device: {data.device}</span>
          <span className="block">Backends: {data.backends.join(", ") || "—"}</span>
          <span className="block">
            Loaded: {data.models_loaded.length > 0 ? data.models_loaded.join(", ") : "none"}
          </span>
        </span>
      }
    >
      <span className="flex cursor-help items-center gap-1.5 text-[11px] text-fg-muted">
        <span className="size-1.5 rounded-full bg-success" />
        <Cpu className="size-3" />
        <span className="font-medium text-fg">{data.device}</span>
      </span>
    </Tooltip>
  );
}

export function TopBar() {
  const source = useAppStore((s) => s.source);
  const clearSource = useAppStore((s) => s.clearSource);
  const analysis = useAppStore((s) => s.analysis);
  const running = useAppStore(selectIsRunning);
  const { theme, toggle } = useTheme();
  const { open, input } = useFilePicker();

  return (
    <header className="flex h-11 shrink-0 items-center justify-between gap-4 border-b border-border bg-panel px-3">
      {input}

      <div className="flex items-center gap-2.5">
        <span className="text-[13px] font-semibold tracking-tight text-fg">
          Vision<span className="text-accent">SR</span>
        </span>
        <span className="h-3.5 w-px bg-border" />
        <BackendStatus />
      </div>

      <div className="flex min-w-0 flex-1 items-center justify-center gap-2 text-[11px] text-fg-subtle">
        {source && (
          <>
            <span className="max-w-[280px] truncate text-fg-muted" title={source.name}>
              {source.name}
            </span>
            <span className="tabular">{formatBytes(source.size)}</span>
            {analysis && (
              <span className="tabular">
                {analysis.width} x {analysis.height}
              </span>
            )}
          </>
        )}
      </div>

      <div className="flex items-center gap-1">
        <Button variant="secondary" size="sm" onClick={open} disabled={running}>
          <FolderOpen />
          Open
        </Button>
        {source && (
          <Tooltip label="Clear the workspace">
            <Button
              variant="ghost"
              size="icon"
              onClick={clearSource}
              disabled={running}
              aria-label="Clear image"
            >
              <Trash2 />
            </Button>
          </Tooltip>
        )}
        <Tooltip label={theme === "dark" ? "Switch to light" : "Switch to dark"}>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggle}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun /> : <Moon />}
          </Button>
        </Tooltip>
      </div>
    </header>
  );
}
