"use client";

import { useEffect } from "react";
import { analyzeImage } from "@/lib/api";
import { errorMessage } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

/**
 * Runs POST /analyze whenever a new source image is loaded, so the user sees
 * what the engine detected *before* committing to an upscale. A failed analysis
 * is non-fatal: it's surfaced in the panel, and the job can still be started.
 */
export function useAnalyzeSource(): void {
  const file = useAppStore((s) => s.source?.file ?? null);
  const setAnalysis = useAppStore((s) => s.setAnalysis);
  const setAnalyzing = useAppStore((s) => s.setAnalyzing);
  const setAnalysisError = useAppStore((s) => s.setAnalysisError);

  useEffect(() => {
    if (!file) return;

    const controller = new AbortController();
    setAnalyzing(true);
    setAnalysisError(null);

    analyzeImage(file, controller.signal)
      .then((analysis) => {
        if (controller.signal.aborted) return;
        setAnalysis(analysis);
        setAnalyzing(false);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setAnalysisError(errorMessage(err));
      });

    return () => controller.abort();
  }, [file, setAnalysis, setAnalyzing, setAnalysisError]);
}
