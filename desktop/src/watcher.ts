/**
 * Hot-folder watching.
 *
 * Drop a photo into the watched folder and it comes back enhanced, next to the
 * original, without anyone opening the app. This is the feature that makes a desktop
 * build worth having over the web one — it is a background service with a window
 * attached, not a page.
 *
 * It runs entirely in the main process and talks to the backend over the same HTTP
 * API the UI uses. Two consequences worth stating: the backend's single-worker queue
 * already serialises these against interactive jobs, so a hot folder cannot starve
 * the user; and a crash in the renderer does not stop the folder being processed.
 *
 * The subtlety is knowing when a file has finished arriving. A large TIFF copied over
 * a network share fires `add` the moment the first byte lands, and reading it then
 * gets a truncated image. chokidar's `awaitWriteFinish` polls the size until it stops
 * changing, which is the only portable way to tell.
 */

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import chokidar, { type FSWatcher } from "chokidar";

export interface WatchEvent {
  kind: "started" | "stopped" | "queued" | "done" | "failed";
  folder?: string;
  file?: string;
  output?: string;
  error?: string;
}

export interface WatchOptions {
  folder: string;
  apiUrl: string;
  scale: number;
  emit: (event: WatchEvent) => void;
}

const IMAGE_SUFFIXES = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff",
]);

/** Marks our own output, so watching a folder does not feed on its own results. */
const OUTPUT_TAG = "_visionsr";

export class HotFolder {
  private watcher: FSWatcher | null = null;

  /** Files already handled, so a touch or a re-scan does not reprocess them. */
  private readonly seen = new Set<string>();

  /**
   * Serialises submissions. The backend queues anyway, but sending fifty files at
   * once would put fifty multi-megabyte uploads in memory simultaneously for no gain.
   */
  private chain: Promise<void> = Promise.resolve();

  async start(options: WatchOptions): Promise<void> {
    await this.stop();

    const watcher = chokidar.watch(options.folder, {
      depth: 0, // the folder itself, not a tree — a hot folder is a drop box
      ignoreInitial: true, // existing files are not "new"; only react to arrivals
      awaitWriteFinish: {
        // A large TIFF copied over a network share fires `add` on its first byte.
        // Reading it then gets a truncated image. Polling the size until it settles
        // is the only portable way to know the copy has finished.
        stabilityThreshold: 1200,
        pollInterval: 200,
      },
    });

    this.watcher = watcher;

    watcher.on("add", (file) => {
      if (!this.shouldProcess(file)) return;

      this.seen.add(file);
      options.emit({ kind: "queued", file, folder: options.folder });

      this.chain = this.chain.then(() => this.process(file, options));
    });

    watcher.on("error", (error) =>
      options.emit({
        kind: "failed",
        folder: options.folder,
        error: error instanceof Error ? error.message : String(error),
      }),
    );

    // Do not report "watching" until chokidar actually is. Its initial scan is async,
    // and with `ignoreInitial` anything that lands before the scan completes is
    // classified as pre-existing and silently skipped. Returning early therefore opens
    // a window — small, but real — in which a file dropped straight after the user
    // picks the folder is dropped on the floor, with no error and no output.
    await new Promise<void>((resolve, reject) => {
      watcher.once("ready", resolve);
      watcher.once("error", reject);
    });

    options.emit({ kind: "started", folder: options.folder });
  }

  async stop(): Promise<void> {
    if (this.watcher === null) return;

    const watcher = this.watcher;
    this.watcher = null;
    this.seen.clear();

    await watcher.close();
  }

  get isWatching(): boolean {
    return this.watcher !== null;
  }

  private shouldProcess(file: string): boolean {
    if (this.seen.has(file)) return false;
    if (!IMAGE_SUFFIXES.has(path.extname(file).toLowerCase())) return false;
    // Our own output lands in the same folder; picking it up would upscale it again,
    // and again, until the disk filled.
    if (path.basename(file, path.extname(file)).endsWith(OUTPUT_TAG)) return false;
    return true;
  }

  private async process(file: string, options: WatchOptions): Promise<void> {
    try {
      const bytes = await readFile(file);

      const form = new FormData();
      form.append("file", new Blob([bytes]), path.basename(file));
      form.append("scale", String(options.scale));
      form.append("output_format", "png");

      const created = await fetch(`${options.apiUrl}/api/v1/jobs`, {
        method: "POST",
        body: form,
      });
      if (!created.ok) {
        throw new Error(`backend refused the job (HTTP ${created.status})`);
      }

      const { job_id: jobId } = (await created.json()) as { job_id: string };
      await this.awaitJob(options.apiUrl, jobId);

      const result = await fetch(`${options.apiUrl}/api/v1/jobs/${jobId}/result`);
      if (!result.ok) {
        throw new Error(`could not fetch the result (HTTP ${result.status})`);
      }

      const output = path.join(
        path.dirname(file),
        `${path.basename(file, path.extname(file))}${OUTPUT_TAG}.png`,
      );
      await writeFile(output, Buffer.from(await result.arrayBuffer()));

      // Remember the output too: chokidar is about to see it appear, and shouldProcess
      // filters it by name, but belt and braces.
      this.seen.add(output);

      options.emit({ kind: "done", file, output, folder: options.folder });
    } catch (error) {
      options.emit({
        kind: "failed",
        file,
        folder: options.folder,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /** Poll rather than stream: this is a background task, latency does not matter. */
  private async awaitJob(apiUrl: string, jobId: string): Promise<void> {
    for (;;) {
      await new Promise((r) => setTimeout(r, 600));

      const response = await fetch(`${apiUrl}/api/v1/jobs/${jobId}`);
      if (!response.ok) {
        throw new Error(`lost track of the job (HTTP ${response.status})`);
      }

      const job = (await response.json()) as { status: string; error: string | null };

      if (job.status === "done") return;
      if (job.status === "failed") {
        throw new Error(job.error ?? "the job failed");
      }
    }
  }
}
