/**
 * Drives the real desktop app.
 *
 * The interesting failures in an Electron shell are not in the UI — that is the same
 * bundle the browser build already exercises. They are in the seams:
 *
 *   - does the Python sidecar actually start, on a port nobody told it in advance?
 *   - does the renderer *find* that port, given the URL is injected at runtime?
 *   - does the preload bridge exist, with only what it should expose?
 *   - and, the one that matters most: **does the engine die when the app does?**
 *     An orphaned uvicorn holding the GPU is invisible, and it makes the next launch
 *     fail for reasons that look nothing like the cause.
 *
 *   node e2e-desktop.mjs
 */
import { _electron as electron } from "playwright";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const SHOTS = path.join(ROOT, "outputs", "e2e-desktop");
const SAMPLE = path.join(ROOT, "outputs", "portrait_input.png");

fs.mkdirSync(SHOTS, { recursive: true });

const failures = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
};

async function backendAlive(url) {
  try {
    const response = await fetch(`${url}/api/v1/health`, { signal: AbortSignal.timeout(2500) });
    return response.ok;
  } catch {
    return false;
  }
}

// ELECTRON_RUN_AS_NODE turns electron.exe into a plain Node interpreter: no GUI, and
// `require("electron")` yields the path to the binary instead of the API, so `app` is
// undefined and the main script dies on its first line. Some CI and agent environments
// export it globally. Strip it, or the app cannot start and the reason looks like a
// bug in the app.
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const app = await electron.launch({ args: ["."], cwd: HERE, env });
let apiUrl = "";

try {
  const page = await app.firstWindow();
  await page.waitForLoadState("domcontentloaded");

  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  check("app launches and opens a window", true);

  // The preload bridge, and the runtime-injected sidecar URL.
  apiUrl = await page.evaluate(() => window.visionsr?.apiUrl ?? "");
  check("preload exposes the sidecar URL", /^http:\/\/127\.0\.0\.1:\d+$/.test(apiUrl), apiUrl);

  check(
    "the sidecar is on an ephemeral port, not a hard-coded one",
    apiUrl !== "" && !apiUrl.endsWith(":8000"),
    apiUrl,
  );

  check("the Python engine is answering", await backendAlive(apiUrl));

  // The bridge must expose the desktop verbs and *not* leak Node.
  const surface = await page.evaluate(() => ({
    keys: Object.keys(window.visionsr ?? {}).sort(),
    leakedRequire: typeof window.require !== "undefined",
    leakedProcess: typeof window.process !== "undefined",
  }));

  check(
    "bridge exposes the desktop API",
    ["openImages", "saveImage", "chooseWatchFolder", "stopWatching", "onWatchEvent"].every((k) =>
      surface.keys.includes(k),
    ),
    surface.keys.join(", "),
  );
  check("node is not leaked into the renderer", !surface.leakedRequire && !surface.leakedProcess);

  // The page must have found the engine through the injected URL — the GPU name only
  // renders if /health came back.
  await page.waitForTimeout(2500);
  const header = await page.locator("body").innerText();
  check("the UI reached the engine", /nvidia|cuda|cpu|rtx/i.test(header));

  await page.screenshot({ path: path.join(SHOTS, "1-launched.png") });

  // Run a real job through the desktop app.
  await page.locator('input[type="file"]').first().setInputFiles(SAMPLE);
  await page.waitForTimeout(2500);

  await page.getByRole("button", { name: /enhance|upscale|run/i }).first().click();

  const done = await page
    .getByText(/download/i)
    .first()
    .waitFor({ timeout: 120_000 })
    .then(() => true)
    .catch(() => false);

  check("a real job runs end to end inside the desktop app", done);

  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(SHOTS, "2-result.png") });

  const body = await page.locator("body").innerText();
  check("result reports the engine that ran it", /Real-ESRGAN/i.test(body));
  check("no uncaught renderer errors", errors.length === 0, errors.slice(0, 2).join(" | "));
} catch (error) {
  check("desktop run", false, error.message);
} finally {
  await app.close();
}

// The whole point. Give the shutdown path a moment, then confirm nothing is left.
await new Promise((r) => setTimeout(r, 4000));

if (apiUrl !== "") {
  const orphaned = await backendAlive(apiUrl);
  check(
    "the Python engine is killed when the app closes",
    !orphaned,
    orphaned ? `${apiUrl} is still answering — the GPU is held by an orphan` : "",
  );
}

console.log(`\nscreenshots -> ${SHOTS}`);
if (failures.length) {
  console.log(`\n${failures.length} check(s) failed: ${failures.join(", ")}`);
  process.exit(1);
}
console.log("\nall checks passed");
