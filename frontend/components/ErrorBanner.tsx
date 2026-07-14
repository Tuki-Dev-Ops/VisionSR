"use client";

import { AlertTriangle, RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useHealth } from "@/hooks/useBackend";
import { API_BASE_URL } from "@/lib/api";
import { useAppStore } from "@/store/appStore";

/**
 * One place for the two failures the user must never miss:
 * the backend being unreachable, and a rejected action (upload/submit).
 */
export function ErrorBanner() {
  const error = useAppStore((s) => s.error);
  const setError = useAppStore((s) => s.setError);
  const { isError: backendDown, refetch, isFetching } = useHealth();

  if (!backendDown && !error) return null;

  return (
    <div className="shrink-0 space-y-px">
      {backendDown && (
        <div className="flex items-center gap-2 border-b border-danger/30 bg-danger-soft px-3 py-1.5 text-[12px] text-danger">
          <AlertTriangle className="size-3.5 shrink-0" />
          <span className="min-w-0 flex-1 truncate">
            Cannot reach the VisionSR backend at{" "}
            <code className="font-mono">{API_BASE_URL}</code>. Start it, then
            retry.
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="text-danger hover:bg-danger/15"
            onClick={() => void refetch()}
            disabled={isFetching}
          >
            <RefreshCw className={isFetching ? "animate-spin" : ""} />
            Retry
          </Button>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 border-b border-danger/30 bg-danger-soft px-3 py-1.5 text-[12px] text-danger">
          <AlertTriangle className="size-3.5 shrink-0" />
          <span className="min-w-0 flex-1">{error}</span>
          <Button
            variant="ghost"
            size="icon"
            className="text-danger hover:bg-danger/15"
            onClick={() => setError(null)}
            aria-label="Dismiss error"
          >
            <X />
          </Button>
        </div>
      )}
    </div>
  );
}
