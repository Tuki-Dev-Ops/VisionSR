/**
 * Surface the Electron preload exposes on `window`.
 *
 * Optional throughout: the same bundle runs in a plain browser, where none of this
 * exists, and every call site has to keep working without it.
 */
export interface VisionSRDesktop {
  /** Base URL of the Python sidecar. The port is chosen at launch, so it is not static. */
  readonly apiUrl: string;
  readonly platform: NodeJS.Platform;
  readonly version: string;

  /** Native open dialog. Returns the chosen files, or [] if cancelled. */
  openImages(): Promise<Array<{ name: string; bytes: ArrayBuffer }>>;

  /** Native save dialog. Returns the path written, or null if cancelled. */
  saveImage(name: string, bytes: ArrayBuffer): Promise<string | null>;

  /** Pick a hot folder to watch. New images in it are enhanced automatically. */
  chooseWatchFolder(): Promise<string | null>;
  stopWatching(): Promise<void>;

  /** Fires as the hot folder makes progress. Returns an unsubscribe function. */
  onWatchEvent(handler: (event: WatchEvent) => void): () => void;
}

export interface WatchEvent {
  kind: "started" | "stopped" | "queued" | "done" | "failed";
  folder?: string;
  file?: string;
  output?: string;
  error?: string;
}

declare global {
  interface Window {
    visionsr?: VisionSRDesktop;
  }
}
