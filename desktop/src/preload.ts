/**
 * The only bridge between the renderer and Node.
 *
 * `contextIsolation` is on and `nodeIntegration` is off, so the page cannot reach
 * `require`, the filesystem, or the child process — it gets exactly the surface
 * declared here and nothing else. That is not ceremony: the renderer loads a bundle
 * and decodes image files, and neither of those should be one bug away from
 * `fs.unlink`.
 *
 * Everything is a plain function returning a plain value. Nothing here holds state,
 * so a reload of the renderer cannot leave the bridge inconsistent.
 */

import { contextBridge, ipcRenderer } from "electron";

export interface WatchEvent {
  kind: "started" | "stopped" | "queued" | "done" | "failed";
  folder?: string;
  file?: string;
  output?: string;
  error?: string;
}

const api = {
  /** Chosen at launch, because the sidecar's port is not known until it starts. */
  apiUrl: process.argv.find((arg) => arg.startsWith("--api-url="))?.slice(10) ?? "",

  platform: process.platform,
  version: process.argv.find((arg) => arg.startsWith("--app-version="))?.slice(14) ?? "",

  async openImages(): Promise<Array<{ name: string; bytes: ArrayBuffer }>> {
    return ipcRenderer.invoke("dialog:open-images");
  },

  /**
   * Write the result somewhere the user chose.
   *
   * The browser build can only trigger a download into the downloads folder. On the
   * desktop, "save this 192MP TIFF next to the original" is the whole point, so this
   * goes through a real save dialog and a real write.
   */
  async saveImage(name: string, bytes: ArrayBuffer): Promise<string | null> {
    return ipcRenderer.invoke("dialog:save-image", name, new Uint8Array(bytes));
  },

  async chooseWatchFolder(): Promise<string | null> {
    return ipcRenderer.invoke("watch:choose-folder");
  },

  async stopWatching(): Promise<void> {
    return ipcRenderer.invoke("watch:stop");
  },

  onWatchEvent(handler: (event: WatchEvent) => void): () => void {
    const listener = (_: unknown, event: WatchEvent) => handler(event);
    ipcRenderer.on("watch:event", listener);
    // Returning the unsubscribe rather than exposing `off` means a caller cannot
    // accidentally remove someone else's listener.
    return () => ipcRenderer.off("watch:event", listener);
  },
} as const;

contextBridge.exposeInMainWorld("visionsr", api);
