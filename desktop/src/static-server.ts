/**
 * Serves the exported frontend over loopback HTTP.
 *
 * The alternative is loading it from `file://`, and that fails for reasons that are
 * tedious to discover one at a time: a Next export references its assets with
 * absolute paths (`/_next/...`), which under `file://` resolve against the drive root;
 * and `file://` is an opaque origin, so `fetch` to the sidecar is a cross-origin
 * request from `null`, which browsers treat with maximum suspicion.
 *
 * A ten-line static server on 127.0.0.1 sidesteps all of it, and the page then lives
 * at a normal `http://` origin behaving exactly as it does in a browser — which is
 * also why the same bundle works in both.
 */

import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import path from "node:path";

const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

export interface StaticServer {
  readonly url: string;
  stop(): Promise<void>;
}

export async function serveStatic(root: string): Promise<StaticServer> {
  const server: Server = createServer(async (request, response) => {
    const requested = decodeURIComponent((request.url ?? "/").split("?")[0]);

    // Refuse to serve outside the export directory. This server only ever talks to
    // our own renderer, but a path-traversal hole is not worth leaving open on the
    // grounds that nobody is expected to walk through it.
    const resolved = path.resolve(root, `.${requested}`);
    if (!resolved.startsWith(path.resolve(root))) {
      response.writeHead(403).end("Forbidden");
      return;
    }

    let file = resolved;
    try {
      const info = await stat(file);
      if (info.isDirectory()) file = path.join(file, "index.html");
    } catch {
      // A Next export writes `index.html` per route; anything else that misses is a
      // 404, not a rewrite to the SPA shell — there is only one route here.
      file = path.join(root, "index.html");
    }

    try {
      await stat(file);
    } catch {
      response.writeHead(404).end("Not found");
      return;
    }

    response.writeHead(200, {
      "Content-Type": CONTENT_TYPES[path.extname(file).toLowerCase()] ?? "application/octet-stream",
      "Cache-Control": "no-store",
    });
    createReadStream(file).pipe(response);
  });

  await new Promise<void>((resolve, reject) => {
    server.on("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });

  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("the static server did not bind to a port");
  }

  return {
    url: `http://127.0.0.1:${address.port}`,
    stop: () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve());
      }),
  };
}
