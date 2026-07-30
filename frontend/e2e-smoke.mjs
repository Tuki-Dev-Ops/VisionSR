/**
 * End-to-end smoke test: drive the real UI against the real backend.
 *
 * Not a unit test and not a replacement for one. This exists because the frontend's
 * whole job is visual, and "it compiles" says nothing about whether a user can drop
 * an image in and get an upscaled one out. It uploads a genuinely degraded photo,
 * waits for the job to finish through the real SSE stream, and screenshots each step.
 *
 *   node e2e-smoke.mjs
 */
import { launchBrowser } from "./e2e-browser.mjs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHOTS = path.join(ROOT, "outputs", "e2e");
const SAMPLE = path.join(ROOT, "outputs", "portrait_input.png");

const APP = process.env.APP_URL ?? "http://127.0.0.1:3000";

fs.mkdirSync(SHOTS, { recursive: true });

const failures = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
};

const browser = await launchBrowser();
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

// Any uncaught client-side error is a failure, even if the UI still looks fine.
const consoleErrors = [];
page.on("pageerror", (e) => consoleErrors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") consoleErrors.push(m.text());
});

try {
  await page.goto(APP, { waitUntil: "networkidle" });
  await page.screenshot({ path: path.join(SHOTS, "1-empty.png") });
  check("app loads", true);

  // The app must have reached the backend — if it did not, everything below is moot.
  const sawBackend = await page
    .getByText(/cuda|rtx|nvidia|cpu/i)
    .first()
    .isVisible()
    .catch(() => false);
  check("backend reachable from the browser", sawBackend);

  // Upload through the real file input rather than synthesising a drop event: this
  // exercises the same code path a user's drag-and-drop lands in.
  const input = page.locator('input[type="file"]').first();
  await input.setInputFiles(SAMPLE);

  await page.waitForTimeout(2500); // let /analyze return
  await page.screenshot({ path: path.join(SHOTS, "2-analyzed.png"), fullPage: true });

  const analysed = await page
    .getByText(/portrait/i)
    .first()
    .isVisible()
    .catch(() => false);
  check("analysis panel shows the detected content type", analysed);

  // Run the job.
  const run = page
    .getByRole("button", { name: /enhance|upscale|run|start/i })
    .first();
  check("run button present", await run.isVisible().catch(() => false));
  await run.click();

  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(SHOTS, "3-running.png"), fullPage: true });

  // Wait for the result. The job is real inference on a 4GB laptop GPU, so allow for it.
  const done = await page
    .getByText(/download|before|after|✓|done/i)
    .first()
    .waitFor({ timeout: 120_000 })
    .then(() => true)
    .catch(() => false);
  check("job completes and the result appears", done);

  await page.waitForTimeout(2000);
  await page.screenshot({ path: path.join(SHOTS, "4-result.png"), fullPage: true });

  // The result panel must report what actually ran. This is what distinguishes a real
  // pipeline from a component that renders a placeholder: a fabricated panel cannot
  // know that this particular run used two chained models on CUDA in three tiles.
  //
  // Read the rendered text rather than hunting for roles — the panel is a definition
  // list, not a table, and a role-based query here tests Playwright's guess at the
  // markup more than it tests the app.
  const body = await page.locator("body").innerText();

  check("result panel reports the backend used", /backend[\s\S]{0,40}(cuda|cpu)/i.test(body));
  check("result panel reports the models used", /Real-ESRGAN/i.test(body));
  check("result panel reports the tile count", /tiles/i.test(body));
  check("download offered", /download/i.test(body));

  // The comparison slider is the feature the product is judged on. Drag its handle
  // and confirm the divider actually moves.
  const stage = page.locator("main, [class*='workspace'], [class*='Workspace']").first();
  const box = await stage.boundingBox();

  const before = await page.screenshot({ clip: box });
  await page.mouse.move(box.x + box.width * 0.42, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.15, box.y + box.height * 0.5, { steps: 15 });
  await page.mouse.up();
  await page.waitForTimeout(400);
  const after = await page.screenshot({ clip: box });

  // Pixels changed => the divider moved. Comparing bytes is crude but it is the one
  // thing that cannot be faked by a component that renders but does nothing.
  check("dragging the divider changes what is shown", !before.equals(after));
  await page.screenshot({ path: path.join(SHOTS, "5-slider-dragged.png"), fullPage: true });

  check("no uncaught client errors", consoleErrors.length === 0, consoleErrors.slice(0, 2).join(" | "));
} catch (error) {
  check("smoke run", false, error.message);
  await page.screenshot({ path: path.join(SHOTS, "error.png") }).catch(() => {});
} finally {
  await browser.close();
}

console.log(`\nscreenshots -> ${SHOTS}`);
if (failures.length) {
  console.log(`\n${failures.length} check(s) failed: ${failures.join(", ")}`);
  process.exit(1);
}
console.log("\nall checks passed");
