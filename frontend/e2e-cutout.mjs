/**
 * End-to-end check for background removal, against the real backend.
 *
 * Companion to e2e-smoke.mjs, which covers the plain upscale path. This one exists
 * because the feature it tests is one a screenshot can lie about: a cutout rendered
 * on a dark canvas and a cutout that never happened look identical. So this does not
 * merely assert that the UI *says* it removed the background — it decodes the result
 * image in the page and counts genuinely non-opaque pixels, then checks that each
 * backdrop actually repaints what is behind them.
 *
 *   APP_URL=http://127.0.0.1:3001 node e2e-cutout.mjs
 */
import { launchBrowser } from "./e2e-browser.mjs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHOTS = path.join(ROOT, "outputs", "e2e-cutout");
const SAMPLE = path.join(ROOT, "outputs", "portrait_input.png");

const APP = process.env.APP_URL ?? "http://127.0.0.1:3001";

fs.mkdirSync(SHOTS, { recursive: true });

const failures = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
};

const browser = await launchBrowser();
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

const consoleErrors = [];
page.on("pageerror", (e) => consoleErrors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") consoleErrors.push(m.text());
});

/* The multipart body the app actually sent — the contract under test.
   Chromium streams a FormData body with a File in it, and Playwright's postData() /
   postDataBuffer() both come back null for those. So record the FormData on the way
   into fetch instead, which is if anything a stricter check: it is the exact object
   the app constructed, field by field. */
await page.addInitScript(() => {
  const original = window.fetch;
  window.__sentForm = null;
  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : input?.url;
    if (init?.body instanceof FormData && String(url).includes("/api/v1/jobs")) {
      window.__sentForm = [...init.body.entries()].map(([k, v]) => [
        k,
        v instanceof File ? `file:${v.name}` : String(v),
      ]);
    }
    return original(input, init);
  };
});

/** True colour of a single rendered pixel, read back out of a real screenshot. */
async function pixelAt(x, y) {
  const shot = await page.screenshot({ clip: { x, y, width: 1, height: 1 } });
  return page.evaluate(async (b64) => {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = 1;
    c.height = 1;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const [r, g, bl] = ctx.getImageData(0, 0, 1, 1).data;
    return { r, g, b: bl };
  }, shot.toString("base64"));
}

try {
  await page.goto(APP, { waitUntil: "networkidle" });

  await page.locator('input[type="file"]').first().setInputFiles(SAMPLE);
  await page.waitForTimeout(2500); // /analyze

  /* ---- the SR dropdown must not offer segmentation models ---- */
  await page.getByRole("combobox", { name: "Model" }).click();
  const srOptions = await page.locator("[role='option']").allInnerTexts();
  await page.keyboard.press("Escape");
  check(
    "SR model dropdown excludes background-removal models",
    !srOptions.some((o) => /IS-Net|U\^?2-Net/i.test(o)),
    `saw: ${srOptions.join(" | ")}`,
  );

  /* ---- turn background removal on ---- */
  const bgGroup = page.getByRole("radiogroup", { name: "Remove background" });
  check("remove-background toggle present", await bgGroup.isVisible());
  await bgGroup.getByRole("radio", { name: "On" }).click();

  const bgModel = page.getByRole("combobox", { name: "Background model" });
  const feather = page.getByRole("slider", { name: "Edge feather" });
  check("background model select appears when On", await bgModel.isVisible());
  check("edge feather slider appears when On", await feather.isVisible());
  check(
    "edge feather defaults to 0",
    (await feather.getAttribute("aria-valuenow")) === "0",
  );

  /* ---- the background dropdown offers exactly the segmentation models ---- */
  await bgModel.click();
  const bgOptions = await page.locator("[role='option']").allInnerTexts();
  await page.keyboard.press("Escape");
  check(
    "background dropdown offers IS-Net and U^2-Net",
    /IS-Net/i.test(bgOptions.join(" ")) && /U\^?2-Net/i.test(bgOptions.join(" ")),
    `saw: ${bgOptions.join(" | ")}`,
  );

  /* ---- JPEG must be unavailable while the result is RGBA ---- */
  await page.getByRole("combobox", { name: "Output format" }).click();
  const jpeg = page.locator("[role='option']").filter({ hasText: /JPEG/i }).first();
  const jpegDisabled = (await jpeg.getAttribute("aria-disabled")) === "true";
  await page.keyboard.press("Escape");
  check("JPEG is disabled when background removal is on", jpegDisabled);

  await page.screenshot({ path: path.join(SHOTS, "1-controls.png"), fullPage: true });

  /* ---- run ---- */
  await page.getByRole("button", { name: /upscale/i }).first().click();

  const done = await page
    .getByText(/RGBA cutout/i)
    .first()
    .waitFor({ timeout: 180_000 })
    .then(() => true)
    .catch(() => false);
  check("job completes and reports an RGBA cutout", done);

  const sent = new Map(await page.evaluate(() => window.__sentForm ?? []));
  check(
    "form carried remove_background=true",
    sent.get("remove_background") === "true",
    [...sent].map(([k, v]) => `${k}=${v}`).join(" "),
  );
  check(
    "form carried background_feather=0 (the default)",
    sent.get("background_feather") === "0",
  );
  check(
    "form omitted background_model_id for Auto",
    !sent.has("background_model_id"),
  );

  await page.waitForTimeout(1200);

  /* ---- the result really has an alpha channel ----
     Decode the rendered <img> and count non-opaque pixels. This is the assertion the
     whole feature rests on: it cannot pass unless real transparency reached the
     browser. Blob URLs are same-origin, so the canvas is not tainted. */
  const alpha = await page.evaluate(() => {
    const img = document.querySelector('img[alt="Enhanced"]');
    if (!img || !img.naturalWidth) return null;
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let clear = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] < 16) clear++;
    const total = c.width * c.height;
    return { total, clear, fraction: clear / total, w: c.width, h: c.height };
  });
  check(
    "result image decodes with real transparent pixels",
    alpha !== null && alpha.fraction > 0.05,
    alpha
      ? `${alpha.w}x${alpha.h}, ${(alpha.fraction * 100).toFixed(1)}% fully transparent`
      : "no image",
  );

  /* ---- result panel reads well with three models ---- */
  const body = await page.locator("body").innerText();
  check("result panel names the segmentation model", /IS-Net|U\^?2-Net/i.test(body));
  check("result panel still names the SR model", /Real-ESRGAN/i.test(body));

  const dl = page.getByRole("link", { name: /download/i }).first();
  const dlName = await dl.getAttribute("download");
  check(
    "download keeps an alpha-capable extension",
    /\.(png|webp)$/i.test(dlName ?? ""),
    dlName ?? "none",
  );

  /* ---- the backdrop picker ---- */
  const picker = page.getByRole("radiogroup", { name: /preview the cutout over/i });
  check("backdrop picker is shown for a transparent result", await picker.isVisible());

  const canvas = page.locator("[aria-label='Before and after comparison']");
  const box = await canvas.boundingBox();

  // Look at the "after" half only.
  const clip = {
    x: box.x + box.width * 0.55,
    y: box.y + box.height * 0.15,
    width: box.width * 0.4,
    height: box.height * 0.7,
  };

  const shots = {};
  for (const name of ["Checkerboard", "White", "Black", "Green screen"]) {
    await picker.getByRole("radio", { name }).click();
    await page.waitForTimeout(250);
    shots[name] = await page.screenshot({ clip });
    await page.screenshot({
      path: path.join(SHOTS, `2-backdrop-${name.split(" ")[0].toLowerCase()}.png`),
    });
  }

  check("white backdrop repaints the cutout area", !shots.White.equals(shots.Checkerboard));
  check("black backdrop repaints the cutout area", !shots.Black.equals(shots.White));
  check("green backdrop repaints the cutout area", !shots["Green screen"].equals(shots.Black));

  /* Sample a real painted pixel at a location that is *known* to be transparent in the
     result — found by decoding the image's alpha and mapping that pixel back to screen
     coordinates, rather than guessing at a corner and hoping it is background.
     Whatever colour is on screen there IS the backdrop showing through. Reading it out
     of a screenshot is the difference between "the CSS says green" and "the user sees
     green": it goes through compositing, the clip-path and the alpha blend. */
  await page.getByRole("radio", { name: "After" }).click(); // full image, no divider
  await page.waitForTimeout(300);

  const probe = await page.evaluate(() => {
    const img = document.querySelector('img[alt="Enhanced"]');
    const r = img.getBoundingClientRect();
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    // Walk inward a little so we never land on a soft matte edge.
    for (let y = 2; y < c.height; y += 3) {
      for (let x = 2; x < c.width; x += 3) {
        if (d[(y * c.width + x) * 4 + 3] !== 0) continue;
        const sx = r.left + (x / c.width) * r.width;
        const sy = r.top + (y / c.height) * r.height;
        if (sx > r.left + 6 && sy > r.top + 6 && sx < r.right - 6 && sy < r.bottom - 6) {
          return { sx, sy, x, y };
        }
      }
    }
    return null;
  });
  check("found a fully transparent pixel to probe", probe !== null);

  const sample = async (name) => {
    await picker.getByRole("radio", { name }).click();
    await page.waitForTimeout(250);
    return pixelAt(probe.sx, probe.sy);
  };

  const green = await sample("Green screen");
  check(
    "green screen actually paints green behind the subject",
    green.g > 130 && green.r < 80 && green.b < 110,
    `rgb(${green.r}, ${green.g}, ${green.b})`,
  );

  const white = await sample("White");
  check(
    "white backdrop actually paints white behind the subject",
    white.r > 240 && white.g > 240 && white.b > 240,
    `rgb(${white.r}, ${white.g}, ${white.b})`,
  );

  const black = await sample("Black");
  check(
    "black backdrop actually paints black behind the subject",
    black.r < 15 && black.g < 15 && black.b < 15,
    `rgb(${black.r}, ${black.g}, ${black.b})`,
  );

  const checker = await sample("Checkerboard");
  check(
    "checkerboard paints a light neutral behind the subject",
    checker.r > 190 && Math.abs(checker.r - checker.g) < 20,
    `rgb(${checker.r}, ${checker.g}, ${checker.b})`,
  );

  /* ---- the checkerboard must stay fixed in screen space under zoom ---- */
  await picker.getByRole("radio", { name: /checkerboard/i }).click();
  await page.waitForTimeout(150);

  const checkerMetrics = async () =>
    page.evaluate(() => {
      const el = document.querySelector(".bg-alpha-checker");
      if (!el) return null;
      const s = getComputedStyle(el);
      return { size: s.backgroundSize, transform: s.transform };
    });

  const before = await checkerMetrics();
  const zoomBefore = await page.locator("text=/^\\d+%$/").first().innerText();
  await canvas.hover();
  await page.mouse.wheel(0, -600); // zoom in hard
  await page.waitForTimeout(300);
  const after = await checkerMetrics();
  const zoomAfter = await page.locator("text=/^\\d+%$/").first().innerText();
  await page.screenshot({ path: path.join(SHOTS, "3-checker-zoomed.png") });

  // The pattern has four gradient layers, so backgroundSize is a four-tuple; what
  // matters is that every layer is still 16px and that the element carries no
  // transform. If the checker rode the zoom, both would change.
  const squaresFixed = (m) =>
    m !== null &&
    m.size.split(",").every((s) => s.trim() === "16px 16px") &&
    (m.transform === "none" || m.transform === "");

  check(
    "zooming actually changed the zoom level (control)",
    zoomBefore !== zoomAfter,
    `${zoomBefore} -> ${zoomAfter}`,
  );
  check(
    "checker squares stay fixed in screen space under zoom",
    squaresFixed(before) && squaresFixed(after) && before.size === after.size,
    after ? `size=[${after.size}] transform=${after.transform}` : "no checker element",
  );

  check("no uncaught client errors", consoleErrors.length === 0, consoleErrors.slice(0, 2).join(" | "));
} catch (error) {
  check("cutout run", false, error.message);
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
