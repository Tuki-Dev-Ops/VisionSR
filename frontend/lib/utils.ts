import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export const ACCEPTED_MIME_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/bmp",
  "image/tiff",
] as const;

export const ACCEPTED_EXTENSIONS = [
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff",
] as const;

export const ACCEPT_ATTRIBUTE = [
  ...ACCEPTED_MIME_TYPES,
  ...ACCEPTED_EXTENSIONS,
].join(",");

export function isAcceptedImage(file: File): boolean {
  const type = file.type.toLowerCase();
  if ((ACCEPTED_MIME_TYPES as readonly string[]).includes(type)) return true;
  // Some browsers report an empty type for .bmp/.tiff drops — fall back to ext.
  const name = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext));
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDimensions(width: number, height: number): string {
  return `${width.toLocaleString()} x ${height.toLocaleString()}`;
}

/** "night_shot.jpeg" + 4 + "png" -> "night_shot_x4.png" */
export function resultFilename(
  sourceName: string,
  scale: number,
  format: string,
): string {
  const base = sourceName.replace(/\.[^./\\]+$/, "") || "image";
  const ext = format === "jpeg" ? "jpg" : format;
  return `${base}_visionsr_x${scale}.${ext}`;
}

/** "denoise_tile" / "face-restore" -> "Denoise tile" / "Face restore" */
export function humanizeStage(stage: string): string {
  const cleaned = stage.replace(/[_-]+/g, " ").trim();
  if (!cleaned) return "Working";
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  return "An unexpected error occurred.";
}
