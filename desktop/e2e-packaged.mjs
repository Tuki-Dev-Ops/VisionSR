/**
 * Drives the *packaged* app — the one a user would actually install.
 *
 * This is a different program from the one `npm start` runs, and the differences are
 * exactly where packaging bugs live:
 *
 *   - it launches `visionsr-server.exe`, a frozen binary with **no Python and no
 *     PyTorch**, instead of the repo's virtualenv;
 *   - it resolves every path against the PyInstaller bundle and the installed
 *     resources, not against a source tree;
 *   - it runs inference on ONNX Runtime / DirectML, not CUDA.
 *
 * A green `npm run e2e` says nothing about any of that. So this test runs the built
 * executable, from its own directory, and enhances a real image through it.
 *
 *   node e2e-packaged.mjs
 */
import { _electron as electron } from "playwright";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const APP = path.join(HERE, "release", "win-unpacked", "VisionSR.exe");
const SHOTS = path.join(ROOT, "outputs", "e2e-packaged");
const SAMPLE = path.join(ROOT, "outputs", "portrait_input.png");

fs.mkdirSync(SHOTS, { recursive: true });

const failures = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
};

if (!fs.existsSync(APP)) {
  console.log(`FAIL  the packaged app is not built — expected ${APP}`);
  console.log("      run: npx electron-builder --dir");
  process.exit(1);
}

// See e2e-desktop.mjs: this variable turns electron.exe into a plain Node interpreter.
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const app = await electron.launch({ executablePath: APP, args: [], env });
let apiUrl = "";

try {
  const page = await app.firstWindow();
  await page.waitForLoadState("domcontentloaded");

  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  check("the packaged app launches", true);

  apiUrl = await page.evaluate(() => window.visionsr?.apiUrl ?? "");
  check("the frozen backend is up", /^http:\/\/127\.0\.0\.1:\d+$/.test(apiUrl), apiUrl);

  // The whole point of the packaging decision: no CUDA, no torch, DirectML instead.
  const health = await page.evaluate(async (url) => {
    const response = await fetch(`${url}/api/v1/health`);
    return response.json();
  }, apiUrl);

  check(
    "it reports a GPU backend (DirectML), not a CPU fallback",
    health.backends.includes("directml"),
    `backends: ${health.backends.join(", ")}`,
  );
  check(
    "PyTorch is genuinely absent from the shipped runtime",
    !health.backends.includes("cuda") && !health.backends.includes("cpu"),
    "cuda/cpu are torch-only backends; their absence proves torch is not bundled",
  );

  // Every model must have found its weights under the installed resources directory.
  const models = await page.evaluate(async (url) => {
    const response = await fetch(`${url}/api/v1/models`);
    return response.json();
  }, apiUrl);

  const installed = models.models.filter((m) => m.installed);
  check(
    "every model resolves its weights inside the install",
    installed.length === models.models.length && installed.length > 0,
    `${installed.length}/${models.models.length} installed`,
  );

  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(SHOTS, "1-launched.png") });

  // ...and now actually enhance something.
  await page.locator('input[type="file"]').first().setInputFiles(SAMPLE);
  await page.waitForTimeout(3000);

  await page.getByRole("button", { name: /enhance|upscale|run/i }).first().click();

  const done = await page
    .getByText(/download/i)
    .first()
    .waitFor({ timeout: 180_000 })
    .then(() => true)
    .catch(() => false);

  check("a real job completes inside the packaged app", done);

  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(SHOTS, "2-result.png") });

  const body = await page.locator("body").innerText();
  check("the result ran on DirectML", /directml/i.test(body), "as reported in the result panel");
  check("no uncaught renderer errors", errors.length === 0, errors.slice(0, 2).join(" | "));
} catch (error) {
  check("packaged run", false, error.message);
} finally {
  await app.close();
}

console.log(`\nscreenshots -> ${SHOTS}`);
if (failures.length) {
  console.log(`\n${failures.length} check(s) failed: ${failures.join(", ")}`);
  process.exit(1);
}
console.log("\nall checks passed");
