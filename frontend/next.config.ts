import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The desktop shell loads these files from a local static server rather than a
  // Next server — there is no SSR to do, the whole app is a client-side workspace,
  // and shipping a Node runtime inside Electron just to serve one static page would
  // be a second thing to keep alive and kill cleanly.
  output: "export",

  // next/image's optimiser is a server feature and cannot run in an export. Nothing
  // here uses it: the before/after canvas deliberately renders raw <img> so the very
  // pixels under comparison are not re-encoded.
  images: { unoptimized: true },
};

export default nextConfig;
