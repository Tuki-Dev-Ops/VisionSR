/**
 * Typed client for the VisionSR backend (API v1).
 *
 * Base URL comes from NEXT_PUBLIC_API_URL and defaults to http://127.0.0.1:8000.
 * Every function throws either `ApiError` (backend responded with a non-2xx)
 * or `NetworkError` (backend unreachable) so callers can always render
 * something meaningful instead of a silent failure.
 */

import {
  ApiError,
  NetworkError,
  type Analysis,
  type CreateJobInput,
  type HealthResponse,
  type Job,
  type JobCreatedResponse,
  type JobEvent,
  type ModelsResponse,
} from "./types";

/**
 * Where the backend lives.
 *
 * In the browser this is baked in at build time from NEXT_PUBLIC_API_URL. The desktop
 * shell cannot work that way: it starts the Python sidecar on whatever port happens to
 * be free, so the address is not known until the app is already running. Electron's
 * preload injects it on `window`, and that wins when present.
 *
 * Resolved once, at module load — the preload script runs before any bundle code, so
 * the value is always there by the time this executes.
 */
function resolveBaseUrl(): string {
  const injected =
    typeof window !== "undefined" ? window.visionsr?.apiUrl : undefined;

  const raw = injected ?? process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
  return raw.replace(/\/+$/, "");
}

export const API_BASE_URL = resolveBaseUrl();

const V1 = `${API_BASE_URL}/api/v1`;

/** Absolute URL for an API path, e.g. apiUrl("/jobs/abc/result"). */
export function apiUrl(path: string): string {
  return `${V1}${path}`;
}

/* --------------------------------------------------------------- internals */

interface ErrorBody {
  detail?: unknown;
  message?: unknown;
  error?: unknown;
}

function extractDetail(body: unknown, fallback: string): string {
  if (typeof body === "string" && body.trim().length > 0) return body.trim();
  if (typeof body !== "object" || body === null) return fallback;

  const { detail, message, error } = body as ErrorBody;
  for (const candidate of [detail, message, error]) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
    // FastAPI validation errors: detail is a list of {loc, msg, type}
    if (Array.isArray(candidate)) {
      const msgs = candidate
        .map((item) =>
          typeof item === "object" && item !== null && "msg" in item
            ? String((item as { msg: unknown }).msg)
            : null,
        )
        .filter((m): m is string => m !== null);
      if (msgs.length > 0) return msgs.join("; ");
    }
  }
  return fallback;
}

async function readError(res: Response): Promise<never> {
  let body: unknown = null;
  try {
    const text = await res.text();
    try {
      body = JSON.parse(text) as unknown;
    } catch {
      body = text;
    }
  } catch {
    body = null;
  }
  const detail = extractDetail(body, `${res.status} ${res.statusText}`);
  throw new ApiError(detail, res.status, detail);
}

/** fetch() that converts connection failures into a typed NetworkError. */
async function request(url: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new NetworkError(
      `Cannot reach the VisionSR backend at ${API_BASE_URL}. Is it running?`,
    );
  }
  if (!res.ok) await readError(res);
  return res;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await request(url, init);
  return (await res.json()) as T;
}

/* ----------------------------------------------------------------- queries */

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>(apiUrl("/health"), {
    signal,
    cache: "no-store",
  });
}

export function getModels(signal?: AbortSignal): Promise<ModelsResponse> {
  return requestJson<ModelsResponse>(apiUrl("/models"), {
    signal,
    cache: "no-store",
  });
}

export function analyzeImage(
  file: File,
  signal?: AbortSignal,
): Promise<Analysis> {
  const form = new FormData();
  form.append("file", file);
  return requestJson<Analysis>(apiUrl("/analyze"), {
    method: "POST",
    body: form,
    signal,
  });
}

/* -------------------------------------------------------------------- jobs */

/** The multipart encoding of the tri-state face-restore control. */
function encodeFaceRestore(mode: CreateJobInput["faceRestore"]): string {
  switch (mode) {
    case "on":
      return "true";
    case "off":
      return "false";
    case "auto":
      return ""; // empty string => let the engine decide
  }
}

export function createJob(
  input: CreateJobInput,
  signal?: AbortSignal,
): Promise<JobCreatedResponse> {
  const form = new FormData();
  form.append("file", input.file);
  form.append("scale", String(input.scale));
  if (input.modelId) form.append("model_id", input.modelId);
  form.append("face_restore", encodeFaceRestore(input.faceRestore));
  form.append("face_restore_weight", String(input.faceRestoreWeight));
  form.append("remove_background", input.removeBackground ? "true" : "false");
  // Omitted entirely => the engine picks the segmentation model.
  if (input.backgroundModelId) {
    form.append("background_model_id", input.backgroundModelId);
  }
  form.append("background_feather", String(input.backgroundFeather));
  form.append("sharpen", String(input.sharpen));
  form.append("output_format", input.outputFormat);

  return requestJson<JobCreatedResponse>(apiUrl("/jobs"), {
    method: "POST",
    body: form,
    signal,
  });
}

export function getJob(jobId: string, signal?: AbortSignal): Promise<Job> {
  return requestJson<Job>(apiUrl(`/jobs/${encodeURIComponent(jobId)}`), {
    signal,
    cache: "no-store",
  });
}

export async function getJobResult(
  jobId: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const res = await request(
    apiUrl(`/jobs/${encodeURIComponent(jobId)}/result`),
    { signal, cache: "no-store" },
  );
  return res.blob();
}

export async function getJobSource(
  jobId: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const res = await request(
    apiUrl(`/jobs/${encodeURIComponent(jobId)}/source`),
    { signal, cache: "no-store" },
  );
  return res.blob();
}

export async function deleteJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<void> {
  await request(apiUrl(`/jobs/${encodeURIComponent(jobId)}`), {
    method: "DELETE",
    signal,
  });
}

/** Direct URLs (useful for <img src> / <a download>). */
export function jobResultUrl(jobId: string): string {
  return apiUrl(`/jobs/${encodeURIComponent(jobId)}/result`);
}

export function jobSourceUrl(jobId: string): string {
  return apiUrl(`/jobs/${encodeURIComponent(jobId)}/source`);
}

/* --------------------------------------------------------------------- SSE */

export interface JobEventHandlers {
  onEvent: (event: JobEvent) => void;
  /** Fired once when the stream breaks before reaching a terminal status. */
  onError: (error: Error) => void;
}

function parseJobEvent(raw: string): JobEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;

  const obj = parsed as Record<string, unknown>;
  const status = obj.status;
  if (
    status !== "queued" &&
    status !== "running" &&
    status !== "done" &&
    status !== "failed"
  ) {
    return null;
  }

  const progress = typeof obj.progress === "number" ? obj.progress : 0;
  return {
    status,
    stage: typeof obj.stage === "string" ? obj.stage : "",
    progress: Math.min(1, Math.max(0, progress)),
    error: typeof obj.error === "string" ? obj.error : null,
  };
}

/**
 * Subscribe to a job's SSE stream. Returns an unsubscribe function.
 * The caller is responsible for falling back to polling if `onError` fires.
 */
export function subscribeToJobEvents(
  jobId: string,
  handlers: JobEventHandlers,
): () => void {
  const url = apiUrl(`/jobs/${encodeURIComponent(jobId)}/events`);
  let closed = false;
  let source: EventSource;

  try {
    source = new EventSource(url);
  } catch {
    handlers.onError(new Error("EventSource is not available"));
    return () => {};
  }

  source.onmessage = (ev: MessageEvent<string>) => {
    const event = parseJobEvent(ev.data);
    if (!event) return;
    handlers.onEvent(event);
    if (event.status === "done" || event.status === "failed") {
      closed = true;
      source.close();
    }
  };

  source.onerror = () => {
    // The server closes the stream on terminal status; that surfaces here as
    // an error too, so only report it if we never saw a terminal event.
    source.close();
    if (!closed) {
      closed = true;
      handlers.onError(new Error("SSE stream failed"));
    }
  };

  return () => {
    closed = true;
    source.close();
  };
}
