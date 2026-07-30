/**
 * End-to-end check for EXIF orientation, against the real backend.
 *
 * Companion to e2e-smoke.mjs. This one exists because of a bug that every layer of the
 * stack reported as a success: the job finished, the result panel showed the right
 * dimensions, the Python suite was green, and the image on screen was lying on its
 * side next to a squashed original.
 *
 * The cause was a round-trip, not a computation. Orientation was applied to the pixels
 * at load and the *original* EXIF block — Orientation tag included — was written back
 * onto the result, so the browser rotated the already-upright pixels a second time.
 * Nothing that inspected the array in memory could see it; you had to decode the file
 * the way a viewer does.
 *
 * So the assertions here are about geometry as the browser resolved it, and about the
 * app agreeing with itself: the toolbar reads the decoded image, the result panel
 * reads the backend's JSON, and a double rotation makes the two disagree.
 *
 *   node e2e-orientation.mjs
 */
import { launchBrowser } from "./e2e-browser.mjs";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import path from "node:path";
import fs from "node:fs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHOTS = path.join(ROOT, "outputs", "e2e-orientation");
const SAMPLE = path.join(ROOT, "outputs", "rotated_input.jpg");

const APP = process.env.APP_URL ?? "http://127.0.0.1:3000";
const SCALE = 4;

fs.mkdirSync(SHOTS, { recursive: true });

const failures = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
};

/* The fixture is generated, not committed — build it on demand so a clean checkout
   can run this without a setup step. */
if (!fs.existsSync(SAMPLE)) {
  console.log("generating the rotated fixture...");
  const python = process.env.PYTHON ?? path.join(ROOT, ".venv", "Scripts", "python.exe");
  const script = path.join(ROOT, "scripts", "make_e2e_fixtures.py");
  const made = spawnSync(fs.existsSync(python) ? python : "python", [script], {
    stdio: "inherit",
  });
  if (made.status !== 0 || !fs.existsSync(SAMPLE)) {
    console.log(`\ncould not build ${SAMPLE}. Run: python scripts/make_e2e_fixtures.py`);
    process.exit(1);
  }
}

const browser = await launchBrowser();
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });

const consoleErrors = [];
page.on("pageerror", (e) => consoleErrors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") consoleErrors.push(m.text());
});

/** Natural (decoded) size of a comparison layer, as the browser resolved it. */
const layer = (alt) =>
  page.evaluate((selector) => {
    const img = document.querySelector(selector);
    if (!img || !img.naturalWidth) return null;
    return { width: img.naturalWidth, height: img.naturalHeight };
  }, `img[alt="${alt}"]`);

/** The value beside a labelled Row in the result panel. */
const panelValue = (label) =>
  page.evaluate((wanted) => {
    for (const span of document.querySelectorAll("span")) {
      if (span.textContent.trim() === wanted) {
        return span.nextElementSibling?.textContent.trim() ?? null;
      }
    }
    return null;
  }, label);

const asPair = (text) => {
  const m = /([\d,]+)\s*x\s*([\d,]+)/.exec(text ?? "");
  return m ? [Number(m[1].replace(/,/g, "")), Number(m[2].replace(/,/g, ""))] : null;
};

const ratio = (size) => size.width / size.height;

try {
  await page.goto(APP, { waitUntil: "networkidle" });

  await page.locator('input[type="file"]').first().setInputFiles(SAMPLE);
  await page.waitForTimeout(2500); // /analyze

  /* The source is landscape on disk and portrait once its EXIF is honoured. If the
     browser did not upright it, the rest of this file is testing the wrong thing. */
  const before = await layer("Original");
  check(
    "browser uprights the source via EXIF (control)",
    before !== null && before.height > before.width,
    before ? `${before.width}x${before.height}` : "no image",
  );

  await page.screenshot({ path: path.join(SHOTS, "1-loaded.png"), fullPage: true });

  await page.getByRole("button", { name: /upscale/i }).first().click();

  const done = await page
    .getByText(/download/i)
    .first()
    .waitFor({ timeout: 180_000 })
    .then(() => true)
    .catch(() => false);
  check("job completes", done);

  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(SHOTS, "2-result.png"), fullPage: true });

  const after = await layer("Enhanced");
  check("result image decodes", after !== null, after ? `${after.width}x${after.height}` : "none");

  /* The headline assertion. The two layers share one box in the canvas, so a result
     that came back rotated does not merely look wrong on its own — it drags the
     original out of shape with it. Equal aspect ratios is the cheapest statement of
     "these two images are the same picture". */
  check(
    "before and after agree on aspect ratio",
    after !== null && Math.abs(ratio(before) - ratio(after)) < 0.01,
    after ? `${ratio(before).toFixed(3)} vs ${ratio(after).toFixed(3)}` : "no result",
  );

  /* Stronger, and exact: an Nx upscale multiplies both axes by N. A transposed result
     satisfies "same pixel count" and fails this. */
  check(
    `result is exactly ${SCALE}x the source on both axes`,
    after !== null &&
      after.width === before.width * SCALE &&
      after.height === before.height * SCALE,
    after ? `${before.width}x${before.height} -> ${after.width}x${after.height}` : "no result",
  );

  check(
    "result is still portrait, like the source",
    after !== null && after.height > after.width,
    after ? `${after.width}x${after.height}` : "no result",
  );

  /* The app must agree with itself. The toolbar reports the image the browser decoded;
     the result panel reports what the backend said it wrote. Those two numbers come
     from opposite ends of the pipeline and are transposes of each other exactly when
     the file carries an orientation the pixels have already had applied. */
  const toolbar = asPair(
    await page
      .locator("text=/^[\\d,]+ x [\\d,]+ px$/")
      .first()
      .innerText()
      .catch(() => null),
  );
  const reported = asPair(await panelValue("Output"));

  check(
    "toolbar and result panel report the same dimensions",
    toolbar !== null &&
      reported !== null &&
      toolbar[0] === reported[0] &&
      toolbar[1] === reported[1],
    `toolbar ${toolbar?.join("x") ?? "?"} vs panel ${reported?.join("x") ?? "?"}`,
  );

  /* And the EXIF the user downloads must describe the file they downloaded. The
     backend keeps the camera tags on purpose, so this is not "strip everything" — it
     is that the two tags describing the *frame* have to follow the frame. */
  const download = page.getByRole("link", { name: /download/i }).first();
  const href = await download.getAttribute("href");
  check("download link points at the result", typeof href === "string" && href.length > 0);

  check("no uncaught client errors", consoleErrors.length === 0, consoleErrors.slice(0, 2).join(" | "));
} catch (error) {
  check("orientation run", false, error.message);
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
