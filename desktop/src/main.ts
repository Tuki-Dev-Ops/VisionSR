/**
 * VisionSR desktop shell.
 *
 * Electron's job here is narrow and it should stay that way: start the Python engine,
 * serve the same web frontend the browser build uses, wire up the things a web page
 * cannot do (native open/save, a hot folder), and — above all — make sure the engine
 * dies when the app does.
 *
 * There is deliberately no second implementation of anything. The desktop app and the
 * web app run identical UI code against an identical HTTP API; the only difference is
 * that here the API happens to be a child process.
 */

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  BrowserWindow,
  Menu,
  app,
  dialog,
  ipcMain,
  shell,
  type MenuItemConstructorOptions,
} from "electron";

import { startSidecar, type Sidecar } from "./sidecar";
import { serveStatic, type StaticServer } from "./static-server";
import { HotFolder, type WatchEvent } from "./watcher";

/** …/desktop/dist/main.js -> the repo root. */
const REPO_ROOT = app.isPackaged
  ? process.resourcesPath
  : path.resolve(__dirname, "..", "..");

const FRONTEND_EXPORT = app.isPackaged
  ? path.join(process.resourcesPath, "frontend")
  : path.join(REPO_ROOT, "frontend", "out");

const IMAGE_FILTERS = [
  { name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"] },
];

let window: BrowserWindow | null = null;
let sidecar: Sidecar | null = null;
let statics: StaticServer | null = null;
const hotFolder = new HotFolder();

// A second instance would spawn a second engine and fight the first for the GPU.
if (!app.requestSingleInstanceLock()) {
  app.quit();
}

app.on("second-instance", () => {
  if (window === null) return;
  if (window.isMinimized()) window.restore();
  window.focus();
});

function emitWatchEvent(event: WatchEvent): void {
  window?.webContents.send("watch:event", event);
}

async function createWindow(apiUrl: string, pageUrl: string): Promise<BrowserWindow> {
  const created = new BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 1000,
    minHeight: 700,
    backgroundColor: "#0a0a0b", // matches the app's dark canvas, so no white flash
    show: false, // revealed once the page has painted, not when it starts loading
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false, // the preload needs ipcRenderer; it exposes nothing else
      additionalArguments: [`--api-url=${apiUrl}`, `--app-version=${app.getVersion()}`],
    },
  });

  created.once("ready-to-show", () => created.show());

  // A link to an external site must not replace the app with a browser page.
  created.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });

  await created.loadURL(pageUrl);
  return created;
}

function buildMenu(): void {
  const template: MenuItemConstructorOptions[] = [
    {
      label: "File",
      submenu: [
        {
          label: "Open Image…",
          accelerator: "CmdOrCtrl+O",
          click: () => window?.webContents.send("menu:open"),
        },
        {
          label: "Watch Folder…",
          accelerator: "CmdOrCtrl+Shift+W",
          click: () => window?.webContents.send("menu:watch"),
        },
        {
          label: "Stop Watching",
          click: async () => {
            await hotFolder.stop();
            emitWatchEvent({ kind: "stopped" });
          },
        },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        // Paste is what lets a screenshot go straight into the app; the renderer
        // already listens for the clipboard `paste` event, so the stock role is enough.
        { role: "paste" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function registerIpc(apiUrl: string): void {
  ipcMain.handle("dialog:open-images", async () => {
    const result = await dialog.showOpenDialog({
      title: "Open image",
      properties: ["openFile", "multiSelections"],
      filters: IMAGE_FILTERS,
    });
    if (result.canceled) return [];

    return Promise.all(
      result.filePaths.map(async (file) => {
        const bytes = await readFile(file);
        // Copy into a fresh ArrayBuffer: a Node Buffer is a view onto a shared pool,
        // and handing that across the bridge would expose unrelated memory.
        return {
          name: path.basename(file),
          bytes: bytes.buffer.slice(
            bytes.byteOffset,
            bytes.byteOffset + bytes.byteLength,
          ) as ArrayBuffer,
        };
      }),
    );
  });

  ipcMain.handle("dialog:save-image", async (_event, name: string, bytes: Uint8Array) => {
    const result = await dialog.showSaveDialog({
      title: "Save enhanced image",
      defaultPath: name,
      filters: IMAGE_FILTERS,
    });
    if (result.canceled || result.filePath === undefined) return null;

    await writeFile(result.filePath, bytes);
    return result.filePath;
  });

  ipcMain.handle("watch:choose-folder", async () => {
    const result = await dialog.showOpenDialog({
      title: "Watch a folder",
      message: "New images dropped here are enhanced automatically.",
      properties: ["openDirectory"],
    });
    if (result.canceled) return null;

    const folder = result.filePaths[0];
    await hotFolder.start({ folder, apiUrl, scale: 4, emit: emitWatchEvent });
    return folder;
  });

  ipcMain.handle("watch:stop", async () => {
    await hotFolder.stop();
    emitWatchEvent({ kind: "stopped" });
  });
}

async function shutdown(): Promise<void> {
  // Order matters: stop feeding the engine work, then stop the engine, then the page
  // server. Killing the engine first would leave the hot folder posting into a void
  // and reporting failures on the way out.
  await hotFolder.stop();
  await sidecar?.stop();
  await statics?.stop();

  sidecar = null;
  statics = null;
}

app.on("window-all-closed", () => {
  // No macOS exception. Leaving a hidden process that owns a GPU is not "staying
  // resident", it is a leak the user cannot see.
  app.quit();
});

app.on("before-quit", (event) => {
  if (sidecar === null && statics === null) return;

  event.preventDefault();
  void shutdown().finally(() => app.quit());
});

// Last resort. If the app is killed hard, at least try not to orphan the engine.
process.on("exit", () => {
  void sidecar?.stop();
});

void app.whenReady().then(async () => {
  try {
    // Serve the page first: it is fast and cannot fail in interesting ways, so if the
    // sidecar then dies we still have somewhere to show the error.
    statics = await serveStatic(FRONTEND_EXPORT);
    sidecar = await startSidecar(REPO_ROOT);

    registerIpc(sidecar.url);
    buildMenu();

    window = await createWindow(sidecar.url, statics.url);
    window.on("closed", () => {
      window = null;
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);

    dialog.showErrorBox(
      "VisionSR could not start",
      `${message}\n\n` +
        `Backend: ${REPO_ROOT}\n` +
        `Frontend: ${FRONTEND_EXPORT}\n\n` +
        "In development, check that the virtualenv exists and that the frontend has " +
        "been exported (npm run build in frontend/).",
    );

    await shutdown();
    app.quit();
  }
});
