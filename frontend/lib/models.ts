/**
 * Rules for reading the model registry.
 *
 * The backend serves every model from one /models list, tagged by `task`. The UI
 * has three different consumers of that list (the SR dropdown, the background
 * dropdown, and the result panel) and each one wants a different slice, so the
 * slicing lives here rather than being re-derived — subtly differently — in each
 * component.
 */

import type { JobResult, ModelInfo } from "./types";

/** Tasks the main model dropdown is allowed to offer. */
export const SUPER_RESOLUTION_TASKS = [
  "super_resolution",
  "anime_restoration",
] as const;

export const BACKGROUND_REMOVAL_TASK = "background_removal";

/**
 * The segmentation models the backend ships today.
 *
 * Only a fallback. `models_used` is authoritative about what ran, but mapping an id
 * to its task needs the registry, and /models can fail (or be served by an older
 * backend) while a job still succeeds. Without this, a failed model fetch would make
 * a transparent result render as an opaque one on a dark background — the exact
 * "looks like nothing happened" failure the checkerboard exists to prevent. Treating
 * a known id as segmentation is the safe way to be wrong.
 */
const KNOWN_BACKGROUND_MODEL_IDS: ReadonlySet<string> = new Set([
  "isnet-general",
  "u2netp",
]);

export function isSuperResolutionModel(model: ModelInfo): boolean {
  return (SUPER_RESOLUTION_TASKS as readonly string[]).includes(model.task);
}

export function isBackgroundRemovalModel(model: ModelInfo): boolean {
  return model.task === BACKGROUND_REMOVAL_TASK;
}

/** Models the main (super-resolution) dropdown may offer. */
export function superResolutionModels(models: readonly ModelInfo[]): ModelInfo[] {
  return models.filter(isSuperResolutionModel);
}

/** Models the background dropdown may offer. */
export function backgroundRemovalModels(models: readonly ModelInfo[]): ModelInfo[] {
  return models.filter(isBackgroundRemovalModel);
}

/**
 * Did this result come back with a real alpha channel?
 *
 * Derived from the job payload, not from the pixels: `models_used` names every model
 * that ran, so a segmentation model in that list means the output is a cutout. That is
 * cheaper and more honest than decoding the image and scanning it for a non-opaque
 * pixel — an image can legitimately be RGBA-encoded and fully opaque, and probing
 * would call that transparent.
 */
export function resultHasAlpha(
  result: JobResult,
  models: readonly ModelInfo[] | undefined,
): boolean {
  return result.models_used.some((id) => {
    const model = models?.find((m) => m.id === id);
    return model
      ? isBackgroundRemovalModel(model)
      : KNOWN_BACKGROUND_MODEL_IDS.has(id);
  });
}
