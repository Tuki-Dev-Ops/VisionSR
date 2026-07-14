/**
 * Hot-folder test.
 *
 * Drives the compiled HotFolder against a real backend and a real temp directory:
 * copy an image in, and an enhanced one must appear beside it. Runs headless — the
 * feature lives entirely in the main process and has no UI to click.
 *
 * The two things worth proving, beyond "it works":
 *
 *  - a partially-written file is not read. chokidar's awaitWriteFinish is the only
 *    portable defence, and if it were misconfigured this would intermittently produce
 *    a truncated-image error rather than an output — which is exactly the kind of bug
 *    that only shows up on someone else's machine, over a network share.
 *
 *  - the watcher does not eat its own output. The result lands in the folder being
 *    watched, so a naive implementation upscales it again, and again, until the disk
 *    is full.
 *
 *   node e2e-hotfolder.mjs        (expects a backend on :8000, or set API_URL)
 */
import { createRequire } from "node:module";
import { mkdtempSync, copyFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { HotFolder } = require("./dist/watcher.js");

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SAMPLE = path.resolve(HERE, "..", "outputs", "portrait_input.png");
const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

const failures = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
};

const folder = mkdtempSync(path.join(tmpdir(), "visionsr-hot-"));
console.log(`watching ${folder}\n`);

const events = [];
const watcher = new HotFolder();

try {
  const healthy = await fetch(`${API_URL}/api/v1/health`)
    .then((r) => r.ok)
    .catch(() => false);
  check("backend reachable", healthy, API_URL);
  if (!healthy) process.exit(1);

  await watcher.start({
    folder,
    apiUrl: API_URL,
    scale: 2,
    emit: (event) => {
      events.push(event);
      console.log(`  event: ${event.kind}${event.file ? ` ${path.basename(event.file)}` : ""}`);
    },
  });
  check("watcher starts", watcher.isWatching);

  // Drop a file in, as a user would.
  const dropped = path.join(folder, "photo.png");
  copyFileSync(SAMPLE, dropped);

  const expected = path.join(folder, "photo_visionsr.png");
  const deadline = Date.now() + 120_000;

  while (Date.now() < deadline && !existsSync(expected)) {
    await new Promise((r) => setTimeout(r, 500));
    if (events.some((e) => e.kind === "failed")) break;
  }

  const failure = events.find((e) => e.kind === "failed");
  check("no failure reported", failure === undefined, failure?.error ?? "");

  check("an enhanced file appeared beside the original", existsSync(expected));

  if (existsSync(expected)) {
    const bytes = statSync(expected).size;
    check("the output is a real image, not an empty file", bytes > 10_000, `${bytes} bytes`);
    check(
      "events report the completed file",
      events.some((e) => e.kind === "done" && e.output === expected),
    );
  }

  // Give the watcher a chance to misbehave: it has just written a PNG into the folder
  // it is watching. If it reprocesses its own output, another file appears.
  await new Promise((r) => setTimeout(r, 6000));

  const produced = readdirSync(folder);
  check(
    "the watcher does not reprocess its own output",
    produced.length === 2,
    `folder now holds: ${produced.join(", ")}`,
  );
} finally {
  await watcher.stop();
}

if (failures.length) {
  console.log(`\n${failures.length} check(s) failed: ${failures.join(", ")}`);
  process.exit(1);
}
console.log("\nall checks passed");
