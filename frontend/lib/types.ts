/**
 * Types mirroring the VisionSR backend API contract (v1).
 * Every shape here corresponds 1:1 to a documented endpoint payload.
 */

/* ------------------------------------------------------------------ health */

export interface HealthResponse {
  status: "ok";
  version: string;
  backends: string[];
  device: string;
  models_loaded: string[];
}

/* ------------------------------------------------------------------ models */

export interface ModelInfo {
  id: string;
  name: string;
  task: string;
  architecture: string;
  scale: number;
  content_types: string[];
  description: string;
  installed: boolean;
  backends: string[];
  precisions: string[];
}

export interface ModelsResponse {
  models: ModelInfo[];
}

/* ---------------------------------------------------------------- analysis */

/** [x, y, width, height] in source-image pixels. */
export type FaceBox = [number, number, number, number];

export interface Analysis {
  width: number;
  height: number;
  megapixels: number;
  has_alpha: boolean;
  content_type: string;
  content_confidence: number;
  noise_level: number;
  blur_level: number;
  compression_level: number;
  quality_score: number;
  faces: FaceBox[];
  recommended_model: string;
  recommended_face_model: string | null;
}

/* -------------------------------------------------------------------- jobs */

export type JobStatus = "queued" | "running" | "done" | "failed";

export interface JobResult {
  width: number;
  height: number;
  duration_ms: number;
  models_used: string[];
  backend: string;
  tiles: number;
  analysis: Analysis;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  error: string | null;
  result: JobResult | null;
}

export interface JobCreatedResponse {
  job_id: string;
  status: "queued";
}

/** Payload of a single Server-Sent Event on /jobs/{id}/events. */
export interface JobEvent {
  status: JobStatus;
  stage: string;
  progress: number;
  error: string | null;
}

/* ---------------------------------------------------------------- settings */

export const SCALES = [2, 4, 8, 16] as const;
export type Scale = (typeof SCALES)[number];

export const OUTPUT_FORMATS = ["png", "jpeg", "webp"] as const;
export type OutputFormat = (typeof OUTPUT_FORMATS)[number];

/**
 * Formats that carry an alpha channel.
 *
 * JPEG does not. Writing an RGBA cutout to JPEG does not error — it silently
 * composites the transparency onto white — so a transparent result must never be
 * offered it. This list is the single source of truth for that rule.
 */
export const ALPHA_OUTPUT_FORMATS = ["png", "webp"] as const satisfies readonly OutputFormat[];

export function formatSupportsAlpha(format: OutputFormat): boolean {
  return (ALPHA_OUTPUT_FORMATS as readonly OutputFormat[]).includes(format);
}

/** Tri-state: "auto" lets the engine decide from the analysis. */
export type FaceRestoreMode = "auto" | "on" | "off";

/** Extra blur on the alpha edge, in pixels. 0 keeps the model's own softness. */
export const MAX_BACKGROUND_FEATHER = 32;

export interface JobSettings {
  scale: Scale;
  /** `null` => "Auto" (server picks via recommended_model). */
  modelId: string | null;
  faceRestore: FaceRestoreMode;
  faceRestoreWeight: number;
  sharpen: number;
  outputFormat: OutputFormat;
  /** Cut the subject out. Makes the result RGBA. */
  removeBackground: boolean;
  /** `null` => "Auto" (server picks the segmentation model). */
  backgroundModelId: string | null;
  /** 0..MAX_BACKGROUND_FEATHER px. */
  backgroundFeather: number;
}

export interface CreateJobInput extends JobSettings {
  file: File;
}

/** How live job updates are currently being received. */
export type ProgressTransport = "sse" | "polling";

/* ------------------------------------------------------------------ errors */

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(message: string, status: number, detail: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** Backend could not be reached at all (DNS, connection refused, CORS, offline). */
export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
  }
}
