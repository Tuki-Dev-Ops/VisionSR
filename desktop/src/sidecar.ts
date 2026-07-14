/**
 * The Python backend, as a child of the desktop app.
 *
 * Three things make this harder than "spawn a process":
 *
 * 1. **The port.** Hard-coding 8000 means the app fails to start whenever anything
 *    else already holds it — including a second copy of itself, or the developer's
 *    own `uvicorn` from a terminal. So a free port is claimed at launch and handed
 *    to the renderer, which is why the frontend resolves its API base URL at runtime.
 *
 * 2. **Readiness.** The process existing is not the same as the server accepting
 *    connections: uvicorn has to import torch, probe CUDA, and load the model
 *    registry first, which takes seconds. Showing the window before that produces a
 *    UI that looks broken. So we poll /health and only then reveal.
 *
 * 3. **Death.** An orphaned uvicorn holding a GPU is the worst failure mode here —
 *    invisible, and it makes the *next* launch fail for reasons that look unrelated.
 *    On Windows a plain SIGTERM does not reliably reach the process tree, so the
 *    kill goes through `taskkill /T`.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { createServer } from "node:net";
import path from "node:path";
import { app } from "electron";

export interface Sidecar {
  readonly url: string;
  stop(): Promise<void>;
}

/** Ask the OS for a port nobody is using, then immediately give it back. */
async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.unref();
    probe.on("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      if (address === null || typeof address === "string") {
        probe.close();
        reject(new Error("could not determine a free port"));
        return;
      }
      const { port } = address;
      probe.close(() => resolve(port));
    });
  });
}

/**
 * Where the Python interpreter is.
 *
 * In development that is the repo's virtualenv. In a packaged build it is a
 * PyInstaller bundle placed next to the app resources — the user does not have
 * Python, and must not need it.
 */
function pythonExecutable(repoRoot: string): { command: string; args: string[] } {
  if (app.isPackaged) {
    const bundled = path.join(
      process.resourcesPath,
      "backend",
      process.platform === "win32" ? "visionsr-server.exe" : "visionsr-server",
    );
    return { command: bundled, args: [] };
  }

  const venv =
    process.platform === "win32"
      ? path.join(repoRoot, ".venv", "Scripts", "python.exe")
      : path.join(repoRoot, ".venv", "bin", "python");

  const command = existsSync(venv) ? venv : "python";

  return {
    command,
    args: ["-m", "uvicorn", "backend.app.main:app", "--log-level", "warning"],
  };
}

async function waitForHealth(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastError = "no response";

  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${url}/api/v1/health`, {
        signal: AbortSignal.timeout(2000),
      });
      if (response.ok) return;
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await new Promise((r) => setTimeout(r, 400));
  }

  throw new Error(
    `The VisionSR backend did not become ready within ${timeoutMs / 1000}s (${lastError}).`,
  );
}

export async function startSidecar(repoRoot: string): Promise<Sidecar> {
  const port = await freePort();
  const url = `http://127.0.0.1:${port}`;

  const { command, args } = pythonExecutable(repoRoot);

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    // Unbuffered, or the backend's logs arrive in one lump when it exits and are
    // useless for diagnosing a startup that never finished.
    PYTHONUNBUFFERED: "1",
    PYTHONIOENCODING: "utf-8",
  };

  if (app.isPackaged) {
    // The frozen backend cannot find its own weights: it resolves paths relative to the
    // PyInstaller bundle, which is a temp directory, and defaults its writable data to
    // LOCALAPPDATA. The installer puts the models beside the app instead, so tell it.
    env.VISIONSR_CHECKPOINT_DIR = path.join(process.resourcesPath, "checkpoints");
  }

  const child: ChildProcess = spawn(
    command,
    [...args, "--host", "127.0.0.1", "--port", String(port)],
    {
      cwd: repoRoot,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    },
  );

  child.stdout?.on("data", (chunk) => process.stdout.write(`[backend] ${chunk}`));
  child.stderr?.on("data", (chunk) => process.stderr.write(`[backend] ${chunk}`));

  let exited = false;
  const exitPromise = new Promise<never>((_, reject) => {
    child.on("exit", (code, signal) => {
      exited = true;
      reject(
        new Error(
          `The VisionSR backend exited before it was ready (code ${code}, signal ${signal}).`,
        ),
      );
    });
    child.on("error", (error) => {
      exited = true;
      reject(new Error(`Could not launch the backend (${command}): ${error.message}`));
    });
  });

  // Race readiness against death: if the process falls over during startup, surface
  // *that* rather than sitting through the full health-check timeout to conclude the
  // same thing 60 seconds later.
  await Promise.race([waitForHealth(url, 60_000), exitPromise]);

  const stop = async (): Promise<void> => {
    if (exited || child.pid === undefined) return;

    if (process.platform === "win32") {
      // /T kills the tree. uvicorn's reloader and torch's worker threads are children,
      // and a bare kill leaves them holding the GPU.
      spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { windowsHide: true });
    } else {
      child.kill("SIGTERM");
    }

    // Give it a moment to go quietly, then stop caring.
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, 3000);
      child.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
  };

  return { url, stop };
}
