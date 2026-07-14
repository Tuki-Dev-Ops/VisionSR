"use client";

import { create } from "zustand";
import { formatSupportsAlpha } from "@/lib/types";
import type {
  Analysis,
  FaceRestoreMode,
  Job,
  JobEvent,
  JobSettings,
  JobStatus,
  OutputFormat,
  ProgressTransport,
  Scale,
} from "@/lib/types";

/** The image the user loaded, plus its blob URL for local preview. */
export interface SourceImage {
  file: File;
  /** Object URL — owned by the store, revoked on replace/clear. */
  url: string;
  name: string;
  size: number;
}

export interface JobState {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  error: string | null;
  result: Job["result"];
  /** Object URL of the fetched result image — owned by the store. */
  resultUrl: string | null;
  transport: ProgressTransport;
  startedAt: number;
  /**
   * The settings this job actually ran with.
   *
   * The controls stay live while a result is on screen, so reading the current
   * settings to describe a finished job is a bug waiting to happen: toggle the
   * format to JPEG after a PNG job and the download button would offer to save the
   * PNG bytes as ".jpg". The job owns its own settings.
   */
  settings: JobSettings;
}

interface AppState {
  settings: JobSettings;
  source: SourceImage | null;
  analysis: Analysis | null;
  analyzing: boolean;
  /** Non-fatal: analysis failed but the job can still run. */
  analysisError: string | null;
  /** Fatal-ish, shown in the top banner (upload rejected, job submit failed). */
  error: string | null;
  job: JobState | null;

  setSetting: <K extends keyof JobSettings>(
    key: K,
    value: JobSettings[K],
  ) => void;
  resetSettings: () => void;

  setSource: (file: File) => void;
  clearSource: () => void;

  setAnalyzing: (analyzing: boolean) => void;
  setAnalysis: (analysis: Analysis | null) => void;
  setAnalysisError: (message: string | null) => void;

  setError: (message: string | null) => void;

  startJob: (jobId: string) => void;
  applyJobEvent: (event: JobEvent) => void;
  applyJob: (job: Job) => void;
  setTransport: (transport: ProgressTransport) => void;
  setResultUrl: (url: string) => void;
  failJob: (message: string) => void;
  clearJob: () => void;
}

export const DEFAULT_SETTINGS: JobSettings = {
  scale: 4 satisfies Scale,
  modelId: null,
  faceRestore: "auto" satisfies FaceRestoreMode,
  faceRestoreWeight: 0.5,
  sharpen: 0,
  outputFormat: "png" satisfies OutputFormat,
  removeBackground: false,
  backgroundModelId: null,
  backgroundFeather: 0,
};

/**
 * Keep the settings internally coherent.
 *
 * A transparent result and a JPEG are mutually exclusive: JPEG has no alpha channel,
 * and encoding a cutout into one silently flattens it onto white. The dropdown also
 * disables JPEG while background removal is on, but that is presentation — enforcing
 * it here means no code path (a restored preset, a future keyboard shortcut) can put
 * the two into a combination the backend would quietly ruin.
 */
function reconcile(settings: JobSettings): JobSettings {
  if (settings.removeBackground && !formatSupportsAlpha(settings.outputFormat)) {
    return { ...settings, outputFormat: "png" };
  }
  return settings;
}

function revoke(url: string | null | undefined): void {
  if (url) URL.revokeObjectURL(url);
}

/**
 * `progress` is reported *per stage*: it runs 0 -> 1 within "analyzing", then
 * again within "upscaling", and so on. So we only guard against a value going
 * backwards inside one stage (out-of-order events); a new stage legitimately
 * restarts the bar. A monotonic guard across the whole job would peg the bar at
 * 100% as soon as the first stage completed.
 */
function nextProgress(
  job: JobState,
  incomingStage: string,
  incomingProgress: number,
): number {
  const stage = incomingStage || job.stage;
  const clamped = Math.min(1, Math.max(0, incomingProgress));
  if (stage !== job.stage) return clamped;
  return Math.max(job.progress, clamped);
}

export const useAppStore = create<AppState>()((set, get) => ({
  settings: { ...DEFAULT_SETTINGS },
  source: null,
  analysis: null,
  analyzing: false,
  analysisError: null,
  error: null,
  job: null,

  setSetting: (key, value) =>
    set((state) => ({
      settings: reconcile({ ...state.settings, [key]: value }),
    })),

  resetSettings: () => set({ settings: { ...DEFAULT_SETTINGS } }),

  setSource: (file) => {
    const { source, job } = get();
    revoke(source?.url);
    revoke(job?.resultUrl);
    set({
      source: {
        file,
        url: URL.createObjectURL(file),
        name: file.name,
        size: file.size,
      },
      analysis: null,
      analyzing: false,
      analysisError: null,
      error: null,
      job: null,
    });
  },

  clearSource: () => {
    const { source, job } = get();
    revoke(source?.url);
    revoke(job?.resultUrl);
    set({
      source: null,
      analysis: null,
      analyzing: false,
      analysisError: null,
      error: null,
      job: null,
    });
  },

  setAnalyzing: (analyzing) => set({ analyzing }),
  setAnalysis: (analysis) => set({ analysis, analysisError: null }),
  setAnalysisError: (message) =>
    set({ analysisError: message, analyzing: false }),

  setError: (message) => set({ error: message }),

  startJob: (jobId) => {
    const state = get();
    revoke(state.job?.resultUrl);
    set({
      error: null,
      job: {
        id: jobId,
        status: "queued",
        stage: "queued",
        progress: 0,
        error: null,
        result: null,
        resultUrl: null,
        transport: "sse",
        startedAt: Date.now(),
        settings: { ...state.settings },
      },
    });
  },

  applyJobEvent: (event) =>
    set((state) => {
      if (!state.job) return state;
      // Never let a late/duplicate event walk a finished job backwards.
      if (state.job.status === "done" || state.job.status === "failed") {
        return state;
      }
      return {
        job: {
          ...state.job,
          status: event.status,
          stage: event.stage || state.job.stage,
          progress: nextProgress(state.job, event.stage, event.progress),
          error: event.error,
        },
      };
    }),

  applyJob: (job) =>
    set((state) => {
      if (!state.job || state.job.id !== job.job_id) return state;
      return {
        job: {
          ...state.job,
          status: job.status,
          stage: job.stage || state.job.stage,
          progress:
            job.status === "done"
              ? 1
              : nextProgress(state.job, job.stage, job.progress),
          error: job.error,
          result: job.result,
        },
        analysis: job.result?.analysis ?? state.analysis,
      };
    }),

  setTransport: (transport) =>
    set((state) =>
      state.job ? { job: { ...state.job, transport } } : state,
    ),

  setResultUrl: (url) =>
    set((state) => {
      if (!state.job) {
        revoke(url);
        return state;
      }
      revoke(state.job.resultUrl);
      return { job: { ...state.job, resultUrl: url } };
    }),

  failJob: (message) =>
    set((state) =>
      state.job
        ? {
            job: {
              ...state.job,
              status: "failed",
              error: message,
            },
          }
        : { error: message },
    ),

  clearJob: () => {
    revoke(get().job?.resultUrl);
    set({ job: null });
  },
}));

/* ---------------------------------------------------------------- selectors */

export const selectIsRunning = (state: AppState): boolean =>
  state.job !== null &&
  (state.job.status === "queued" || state.job.status === "running");

export const selectHasResult = (state: AppState): boolean =>
  state.job?.status === "done" && state.job.resultUrl !== null;
