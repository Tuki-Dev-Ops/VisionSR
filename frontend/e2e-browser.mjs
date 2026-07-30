/**
 * One way to get a browser for the end-to-end scripts.
 *
 * Playwright's bundled headless Chromium is the default and stays the default — it is
 * pinned, so a machine that has it gives every run the same engine. But it is a
 * stripped build, and on some Windows hosts it dies at startup with
 *
 *     [ERROR:base\i18n\icu_util.cc] Invalid file descriptor to ICU data received.
 *
 * before Playwright can attach, which surfaces as the unhelpful "Target page, context
 * or browser has been closed". That is a broken *install*, not a broken app, and it
 * should not read as a product failure — so we fall back to a real Chrome or Edge,
 * which every Windows box already has, and say loudly which engine actually ran.
 *
 * Pin one explicitly with PW_CHANNEL=chrome|msedge|chromium.
 */
import { chromium } from "playwright";

/** Bundled build first, then the browsers a desktop is likely to already have. */
const FALLBACKS = [undefined, "chrome", "msedge"];

const name = (channel) => channel ?? "bundled chromium";

/**
 * Launch the first browser that actually starts.
 *
 * @param {import("playwright").LaunchOptions} [options]
 * @returns {Promise<import("playwright").Browser>}
 */
export async function launchBrowser(options = {}) {
  const pinned = process.env.PW_CHANNEL;
  const candidates = pinned
    ? [pinned === "chromium" ? undefined : pinned]
    : FALLBACKS;

  const failures = [];
  for (const channel of candidates) {
    try {
      const browser = await chromium.launch({ ...options, channel });
      if (channel !== undefined) {
        console.log(`note: bundled Chromium unavailable — running on ${channel}`);
      }
      return browser;
    } catch (error) {
      failures.push(`${name(channel)}: ${error.message.split("\n")[0]}`);
    }
  }

  throw new Error(
    `no usable browser (tried ${candidates.map(name).join(", ")}).\n` +
      `  ${failures.join("\n  ")}\n` +
      `Install one with: npx playwright install chromium`,
  );
}
