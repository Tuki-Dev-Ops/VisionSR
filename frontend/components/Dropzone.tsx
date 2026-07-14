"use client";

import * as React from "react";
import { ImagePlus, Upload } from "lucide-react";
import { ACCEPT_ATTRIBUTE, cn, isAcceptedImage } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

/** Validates and accepts a file into the store, or reports why it was rejected. */
export function useImageIntake(): (file: File | null | undefined) => void {
  const setSource = useAppStore((s) => s.setSource);
  const setError = useAppStore((s) => s.setError);

  return React.useCallback(
    (file: File | null | undefined) => {
      if (!file) return;
      if (!isAcceptedImage(file)) {
        setError(
          `"${file.name || "That file"}" is not a supported image. Use PNG, JPG, WebP, BMP or TIFF.`,
        );
        return;
      }
      setSource(file);
    },
    [setSource, setError],
  );
}

/** Paste an image from the clipboard anywhere in the app. */
export function useClipboardPaste(enabled: boolean): void {
  const accept = useImageIntake();

  React.useEffect(() => {
    if (!enabled) return;
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.kind !== "file") continue;
        const file = item.getAsFile();
        if (file) {
          e.preventDefault();
          accept(file);
          return;
        }
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [enabled, accept]);
}

/** Drop an image anywhere in the window (also when one is already loaded). */
export function useWindowDrop(enabled: boolean): boolean {
  const accept = useImageIntake();
  const [dragging, setDragging] = React.useState(false);
  const depth = React.useRef(0);

  React.useEffect(() => {
    if (!enabled) {
      setDragging(false);
      return;
    }
    const hasFiles = (e: DragEvent): boolean =>
      Array.from(e.dataTransfer?.types ?? []).includes("Files");

    const onDragEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current += 1;
      setDragging(true);
    };
    const onDragOver = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
    };
    const onDragLeave = () => {
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setDragging(false);
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current = 0;
      setDragging(false);
      accept(e.dataTransfer?.files?.[0]);
    };

    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [enabled, accept]);

  return dragging;
}

/** Hidden input + click-to-browse trigger, reusable from anywhere. */
export function useFilePicker(): {
  open: () => void;
  input: React.ReactElement;
} {
  const accept = useImageIntake();
  const ref = React.useRef<HTMLInputElement>(null);

  const open = React.useCallback(() => ref.current?.click(), []);

  const input = (
    <input
      ref={ref}
      type="file"
      accept={ACCEPT_ATTRIBUTE}
      className="hidden"
      onChange={(e) => {
        accept(e.target.files?.[0]);
        e.target.value = "";
      }}
    />
  );

  return { open, input };
}

/* ---------------------------------------------------------------- empty UI */

export function Dropzone({ dragging }: { dragging: boolean }) {
  const { open, input } = useFilePicker();

  return (
    <button
      type="button"
      onClick={open}
      className={cn(
        "group flex h-full w-full flex-col items-center justify-center gap-4 rounded-lg border border-dashed transition-colors",
        dragging
          ? "border-accent bg-accent-soft"
          : "border-border bg-panel hover:border-border-strong hover:bg-panel-raised",
      )}
    >
      {input}
      <div
        className={cn(
          "flex size-12 items-center justify-center rounded-full border border-border bg-panel-inset text-fg-subtle transition-colors",
          dragging
            ? "border-accent text-accent"
            : "group-hover:border-border-strong group-hover:text-fg-muted",
        )}
      >
        {dragging ? (
          <Upload className="size-5" />
        ) : (
          <ImagePlus className="size-5" />
        )}
      </div>
      <div className="space-y-1 text-center">
        <p className="text-[13px] text-fg">
          {dragging ? "Drop to load" : "Drop an image, paste, or click to browse"}
        </p>
        <p className="text-[11px] text-fg-subtle">
          PNG &middot; JPG &middot; WebP &middot; BMP &middot; TIFF
        </p>
      </div>
    </button>
  );
}

/** Full-window tint while a file is being dragged over an existing image. */
export function DropOverlay() {
  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center rounded-lg border-2 border-dashed border-accent bg-accent-soft backdrop-blur-[1px]">
      <div className="flex items-center gap-2 rounded-md border border-border bg-panel px-3 py-2 text-[13px] text-fg shadow-xl">
        <Upload className="size-4 text-accent" />
        Drop to replace the image
      </div>
    </div>
  );
}
