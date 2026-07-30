"use client";

/* next/image is deliberately not used here: this canvas needs the raw, untouched
   pixels of both images (no re-encoding, no responsive resizing) and drives its
   own transform/zoom/pan. Both sources are blob: URLs from the local job. */
/* eslint-disable @next/next/no-img-element */

import * as React from "react";
import {
  Maximize2,
  Minus,
  MoveHorizontal,
  Plus,
  Scan,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Segmented, type SegmentedOption } from "@/components/ui/segmented";
import { Tooltip } from "@/components/ui/tooltip";
import { clamp, cn } from "@/lib/utils";

/* ------------------------------------------------------------------ types */

type ViewMode = "split" | "before" | "after";

/** What a transparent result is composited over, for preview only. */
type Backdrop = "checker" | "white" | "black" | "green";

interface View {
  /** Screen pixels per image pixel. 1 = 1:1. */
  zoom: number;
  /** Offset of the image centre from the viewport centre, in screen px. */
  x: number;
  y: number;
}

interface Size {
  width: number;
  height: number;
}

/** Position of the image on screen, in container-local px. */
interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface CompareCanvasProps {
  beforeSrc: string;
  /** null while there is no result yet — the canvas then acts as a viewer. */
  afterSrc: string | null;
  /**
   * The result carries an alpha channel (a cutout).
   *
   * Turns on the checkerboard and the backdrop picker for the "after" side. Without
   * it, a transparent result drawn on the near-black canvas is indistinguishable from
   * one where nothing happened.
   */
  afterHasAlpha?: boolean;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
  /** Dimmed + non-interactive overlay content (e.g. the progress panel). */
  overlay?: React.ReactNode;
}

const MIN_ZOOM = 0.02;
const MAX_ZOOM = 16;
const DIVIDER_HIT_PX = 14;
const ZOOM_STEP = 1.25;

const VIEW_MODES: readonly SegmentedOption<ViewMode>[] = [
  { value: "before", label: "Before", title: "Show the original only" },
  { value: "split", label: "Split", title: "Split comparison" },
  { value: "after", label: "After", title: "Show the result only" },
];

/**
 * A cutout is only as good as it looks against the thing it will be placed on, so the
 * preview offers the four backgrounds that actually diagnose it: the checkerboard
 * reads as "transparent" at a glance, white and black expose opposite halo colours
 * along the matte edge, and green is the composite every video tool will key against.
 */
const BACKDROPS: readonly {
  value: Backdrop;
  label: string;
  title: string;
  /** The swatch preview, and (except for checker) the layer's own paint. */
  swatch: string;
}[] = [
  {
    value: "checker",
    label: "Checkerboard",
    title: "Show transparency as a checkerboard",
    swatch: "",
  },
  { value: "white", label: "White", title: "Composite over white", swatch: "#ffffff" },
  { value: "black", label: "Black", title: "Composite over black", swatch: "#000000" },
  {
    value: "green",
    label: "Green screen",
    title: "Composite over a green screen",
    swatch: "#00b140",
  },
];

/* ------------------------------------------------------------- component */

export function CompareCanvas({
  beforeSrc,
  afterSrc,
  afterHasAlpha = false,
  beforeLabel = "Original",
  afterLabel = "Enhanced",
  className,
  overlay,
}: CompareCanvasProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);

  const [container, setContainer] = React.useState<Size>({
    width: 0,
    height: 0,
  });
  const [beforeSize, setBeforeSize] = React.useState<Size | null>(null);
  const [afterSize, setAfterSize] = React.useState<Size | null>(null);
  const [loadError, setLoadError] = React.useState<string | null>(null);

  const [view, setView] = React.useState<View>({ zoom: 1, x: 0, y: 0 });
  const [fitted, setFitted] = React.useState(true);
  const [divider, setDivider] = React.useState(0.5);
  const [mode, setMode] = React.useState<ViewMode>("split");
  const [nearDivider, setNearDivider] = React.useState(false);
  const [panning, setPanning] = React.useState(false);
  const [backdrop, setBackdrop] = React.useState<Backdrop>("checker");

  /* Both layers are drawn at the *result* resolution so the two images stay
     pixel-aligned under the divider. Before a result exists, the source's own
     size is the display box. */
  const display: Size | null = afterSize ?? beforeSize;

  const fitZoom = React.useMemo(() => {
    if (!display || !container.width || !container.height) return 1;
    return Math.min(
      container.width / display.width,
      container.height / display.height,
    );
  }, [display, container]);

  /* ----------------------------------------------------------- observers */

  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setContainer({ width: rect.width, height: rect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  /* Keep the framing when the display box changes (i.e. the result arrives at
     N× the source resolution): rescale zoom by the same factor so the visible
     region is untouched, or just re-fit if the user never zoomed. */
  const prevDisplay = React.useRef<Size | null>(null);
  React.useEffect(() => {
    const prev = prevDisplay.current;
    prevDisplay.current = display;
    if (!display) return;
    if (!prev || prev.width === display.width) return;
    const ratio = prev.width / display.width;
    setView((v) => ({ ...v, zoom: v.zoom * ratio }));
  }, [display]);

  /* Snap to fit whenever we're in "fitted" mode and the geometry changes. */
  React.useEffect(() => {
    if (!fitted || !display || !container.width) return;
    setView({ zoom: fitZoom, x: 0, y: 0 });
  }, [fitted, fitZoom, display, container.width, container.height]);

  /* The result was cleared (re-run, or a new job started): drop its geometry. */
  React.useEffect(() => {
    if (afterSrc === null) {
      setAfterSize(null);
      setMode("split");
    }
    setLoadError(null);
  }, [afterSrc]);

  /* ------------------------------------------------------------- helpers */

  /** Clamp the pan so the image never detaches from the viewport edges. */
  const clampView = React.useCallback(
    (next: View): View => {
      if (!display || !container.width) return next;
      const w = display.width * next.zoom;
      const h = display.height * next.zoom;
      const limitX = Math.max(0, (w - container.width) / 2);
      const limitY = Math.max(0, (h - container.height) / 2);
      return {
        zoom: next.zoom,
        x: clamp(next.x, -limitX, limitX),
        y: clamp(next.y, -limitY, limitY),
      };
    },
    [display, container],
  );

  /** Zoom about a point given in container-local coordinates. */
  const zoomAbout = React.useCallback(
    (nextZoom: number, px: number, py: number) => {
      if (!display) return;
      setFitted(false);
      setView((v) => {
        const z2 = clamp(nextZoom, MIN_ZOOM, MAX_ZOOM);
        const cx = container.width / 2;
        const cy = container.height / 2;
        // Image-space point currently under (px, py).
        const topLeftX = cx + v.x - (display.width * v.zoom) / 2;
        const topLeftY = cy + v.y - (display.height * v.zoom) / 2;
        const ix = (px - topLeftX) / v.zoom;
        const iy = (py - topLeftY) / v.zoom;
        // Solve for the offset that keeps (ix, iy) under the cursor at z2.
        const x = px - ix * z2 + (display.width * z2) / 2 - cx;
        const y = py - iy * z2 + (display.height * z2) / 2 - cy;
        return clampView({ zoom: z2, x, y });
      });
    },
    [display, container, clampView],
  );

  const zoomBy = React.useCallback(
    (factor: number) => {
      zoomAbout(view.zoom * factor, container.width / 2, container.height / 2);
    },
    [zoomAbout, view.zoom, container],
  );

  const fitToView = React.useCallback(() => {
    setFitted(true);
    setView({ zoom: fitZoom, x: 0, y: 0 });
  }, [fitZoom]);

  const zoomTo100 = React.useCallback(() => {
    setFitted(false);
    setView((v) => clampView({ ...v, zoom: 1 }));
  }, [clampView]);

  const isFit = Math.abs(view.zoom - fitZoom) < 0.001;

  /* --------------------------------------------------------------- wheel */

  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      // Normalise line/page deltas to something pixel-ish.
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1;
      const delta = e.deltaY * unit;
      const factor = Math.exp(-delta * 0.0018);
      zoomAbout(
        view.zoom * factor,
        e.clientX - rect.left,
        e.clientY - rect.top,
      );
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAbout, view.zoom]);

  /* ------------------------------------------------------------ pointers */

  type Gesture =
    | { kind: "none" }
    | { kind: "divider" }
    | { kind: "pan"; startX: number; startY: number; origin: View }
    | {
        kind: "pinch";
        startDist: number;
        startZoom: number;
        startMid: { x: number; y: number };
      };

  const gesture = React.useRef<Gesture>({ kind: "none" });
  const pointers = React.useRef<Map<number, { x: number; y: number }>>(
    new Map(),
  );

  const localPoint = (e: React.PointerEvent): { x: number; y: number } => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const dividerX = divider * container.width;
  const splitActive = mode === "split" && afterSrc !== null;

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!display) return;
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const p = localPoint(e);
    pointers.current.set(e.pointerId, p);

    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      gesture.current = {
        kind: "pinch",
        startDist: Math.hypot(a.x - b.x, a.y - b.y) || 1,
        startZoom: view.zoom,
        startMid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
      };
      setPanning(false);
      return;
    }

    if (splitActive && Math.abs(p.x - dividerX) <= DIVIDER_HIT_PX) {
      gesture.current = { kind: "divider" };
      setDivider(clamp(p.x / container.width, 0, 1));
      return;
    }

    gesture.current = {
      kind: "pan",
      startX: p.x,
      startY: p.y,
      origin: view,
    };
    setPanning(true);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const p = localPoint(e);
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, p);

    const g = gesture.current;

    if (g.kind === "none") {
      setNearDivider(
        splitActive && Math.abs(p.x - dividerX) <= DIVIDER_HIT_PX,
      );
      return;
    }

    if (g.kind === "divider") {
      setDivider(clamp(p.x / container.width, 0, 1));
      return;
    }

    if (g.kind === "pan") {
      setFitted(false);
      setView(
        clampView({
          zoom: g.origin.zoom,
          x: g.origin.x + (p.x - g.startX),
          y: g.origin.y + (p.y - g.startY),
        }),
      );
      return;
    }

    if (g.kind === "pinch" && pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y) || 1;
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      zoomAbout((dist / g.startDist) * g.startZoom, mid.x, mid.y);
    }
  };

  const endPointer = (e: React.PointerEvent<HTMLDivElement>) => {
    pointers.current.delete(e.pointerId);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (pointers.current.size === 0) {
      gesture.current = { kind: "none" };
      setPanning(false);
    } else if (gesture.current.kind === "pinch") {
      gesture.current = { kind: "none" };
    }
  };

  const handleDoubleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (isFit) {
      zoomAbout(1, e.clientX - rect.left, e.clientY - rect.top);
    } else {
      fitToView();
    }
  };

  /* ------------------------------------------------------------ keyboard */

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 0.005 : 0.02;
    switch (e.key) {
      case "ArrowLeft":
        if (!splitActive) return;
        e.preventDefault();
        setDivider((d) => clamp(d - step, 0, 1));
        break;
      case "ArrowRight":
        if (!splitActive) return;
        e.preventDefault();
        setDivider((d) => clamp(d + step, 0, 1));
        break;
      case "+":
      case "=":
        e.preventDefault();
        zoomBy(ZOOM_STEP);
        break;
      case "-":
      case "_":
        e.preventDefault();
        zoomBy(1 / ZOOM_STEP);
        break;
      case "0":
        e.preventDefault();
        fitToView();
        break;
      case "1":
        e.preventDefault();
        zoomTo100();
        break;
      default:
        break;
    }
  };

  /* ------------------------------------------------------------ geometry */

  const layerStyle = (): React.CSSProperties => {
    if (!display) return { display: "none" };
    const w = display.width * view.zoom;
    const h = display.height * view.zoom;
    return {
      position: "absolute",
      left: 0,
      top: 0,
      width: `${display.width}px`,
      height: `${display.height}px`,
      // Both layers share the result's box so they stay aligned under the divider,
      // which for a plain N-times upscale is exactly a scale. If the two ever
      // disagree on aspect ratio, "contain" letterboxes the odd one out instead of
      // silently stretching it — a distorted original reads as a bad result.
      objectFit: "contain",
      transformOrigin: "0 0",
      transform: `translate3d(${container.width / 2 + view.x - w / 2}px, ${
        container.height / 2 + view.y - h / 2
      }px, 0) scale(${view.zoom})`,
      // Above 2x, show the real pixels instead of the browser's smoothing —
      // both layers get identical treatment so the comparison stays honest.
      imageRendering: view.zoom >= 2 ? "pixelated" : "auto",
      willChange: "transform",
    };
  };

  /** Where the image currently sits on screen — the same box layerStyle() paints into. */
  const imageRect: Rect | null =
    display && container.width
      ? {
          left: container.width / 2 + view.x - (display.width * view.zoom) / 2,
          top: container.height / 2 + view.y - (display.height * view.zoom) / 2,
          width: display.width * view.zoom,
          height: display.height * view.zoom,
        }
      : null;

  /**
   * The backdrop the transparent result is composited over.
   *
   * Deliberately *not* a transformed child of the image layer. The checkerboard is a
   * unit of the screen, not of the image: if it rode the zoom transform its squares
   * would balloon to the size of the viewport at 16x and shrink to moire at 2%, which
   * reads as a rendering bug rather than as "this area is empty". So the element spans
   * the whole viewport — anchoring the 16px pattern to the container, where it stays
   * put under pan and zoom — and is then clipped down to the image's rectangle, so it
   * still ends exactly where the result's pixels end.
   */
  const backdropStyle = (): React.CSSProperties => {
    if (!imageRect) return { display: "none" };
    const right = Math.max(0, container.width - (imageRect.left + imageRect.width));
    const bottom = Math.max(0, container.height - (imageRect.top + imageRect.height));
    const inset = [
      Math.max(0, imageRect.top),
      right,
      bottom,
      Math.max(0, imageRect.left),
    ]
      .map((v) => `${v}px`)
      .join(" ");

    const paint = BACKDROPS.find((b) => b.value === backdrop)?.swatch ?? "";
    return {
      position: "absolute",
      inset: 0,
      clipPath: `inset(${inset})`,
      ...(paint ? { background: paint } : null),
    };
  };

  const clipPath =
    mode === "before"
      ? "inset(0 0 0 100%)"
      : mode === "after"
        ? "inset(0 0 0 0)"
        : `inset(0 0 0 ${divider * 100}%)`;

  const cursor = nearDivider
    ? "ew-resize"
    : panning
      ? "grabbing"
      : "grab";

  const ready = display !== null;

  return (
    <div
      className={cn(
        "relative isolate flex h-full w-full flex-col overflow-hidden rounded-lg border border-border bg-panel",
        className,
      )}
    >
      {/* toolbar */}
      <div className="flex items-center justify-between gap-3 border-b border-border bg-panel px-2 py-1.5">
        <div className="w-[210px] shrink-0">
          {afterSrc && (
            <Segmented
              value={mode}
              options={VIEW_MODES}
              onChange={setMode}
              size="sm"
              ariaLabel="Comparison view mode"
            />
          )}
        </div>

        <div className="flex min-w-0 items-center gap-2.5 text-[11px] text-fg-subtle">
          {display && (
            <span className="tabular truncate">
              {display.width.toLocaleString()} x{" "}
              {display.height.toLocaleString()} px
            </span>
          )}

          {/* Backdrop picker — only meaningful when there is transparency to show. */}
          {afterSrc && afterHasAlpha && mode !== "before" && (
            <div
              role="radiogroup"
              aria-label="Preview the cutout over"
              className="flex shrink-0 items-center gap-1 rounded-md border border-border bg-panel-inset p-0.5"
            >
              {BACKDROPS.map((option) => {
                const active = backdrop === option.value;
                return (
                  <Tooltip key={option.value} label={option.title}>
                    <button
                      type="button"
                      role="radio"
                      aria-checked={active}
                      aria-label={option.label}
                      onClick={() => setBackdrop(option.value)}
                      className={cn(
                        "size-5 rounded-[4px] ring-1 transition-transform ring-inset",
                        "hover:scale-110",
                        option.value === "checker" && "bg-alpha-checker-swatch",
                        active
                          ? "ring-accent ring-2"
                          : "ring-border-strong",
                      )}
                      style={
                        option.swatch ? { background: option.swatch } : undefined
                      }
                    />
                  </Tooltip>
                );
              })}
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <Tooltip label="Zoom out (-)">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => zoomBy(1 / ZOOM_STEP)}
              disabled={!ready}
              aria-label="Zoom out"
            >
              <Minus />
            </Button>
          </Tooltip>
          <span className="tabular w-12 text-center text-[11px] text-fg-muted">
            {Math.round(view.zoom * 100)}%
          </span>
          <Tooltip label="Zoom in (+)">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => zoomBy(ZOOM_STEP)}
              disabled={!ready}
              aria-label="Zoom in"
            >
              <Plus />
            </Button>
          </Tooltip>
          <div className="mx-1 h-4 w-px bg-border" />
          <Tooltip label="Fit to view (0)">
            <Button
              variant={isFit ? "primary" : "ghost"}
              size="icon"
              onClick={fitToView}
              disabled={!ready}
              aria-label="Fit to view"
              aria-pressed={isFit}
            >
              <Scan />
            </Button>
          </Tooltip>
          <Tooltip label="Actual pixels, 1:1 (1)">
            <Button
              variant={
                Math.abs(view.zoom - 1) < 0.001 && !isFit ? "primary" : "ghost"
              }
              size="sm"
              onClick={zoomTo100}
              disabled={!ready}
              aria-label="Zoom to 100 percent"
            >
              1:1
            </Button>
          </Tooltip>
          <Tooltip label="Reset divider to centre">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setDivider(0.5)}
              disabled={!splitActive}
              aria-label="Centre the divider"
            >
              <MoveHorizontal />
            </Button>
          </Tooltip>
        </div>
      </div>

      {/* canvas */}
      <div
        ref={containerRef}
        tabIndex={0}
        role="group"
        aria-label="Before and after comparison"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endPointer}
        onPointerCancel={endPointer}
        onPointerLeave={() => setNearDivider(false)}
        onDoubleClick={handleDoubleClick}
        onKeyDown={handleKeyDown}
        className="bg-checker relative min-h-0 flex-1 touch-none overflow-hidden outline-none select-none"
        style={{ cursor: ready ? cursor : "default" }}
      >
        {/* before layer */}
        <img
          src={beforeSrc}
          alt="Original"
          draggable={false}
          onLoad={(e) => {
            const img = e.currentTarget;
            setBeforeSize({
              width: img.naturalWidth,
              height: img.naturalHeight,
            });
          }}
          onError={() => setLoadError("The original image could not be loaded.")}
          style={layerStyle()}
        />

        {/* after layer, clipped in screen space */}
        {afterSrc && (
          <div
            className="pointer-events-none absolute inset-0"
            style={{ clipPath }}
          >
            {/* Sits inside the same clip as the result, so in split mode the backdrop
                stops at the divider and the original keeps its own background. */}
            {afterHasAlpha && (
              <div
                aria-hidden
                className={cn(backdrop === "checker" && "bg-alpha-checker")}
                style={backdropStyle()}
              />
            )}
            <img
              src={afterSrc}
              alt="Enhanced"
              draggable={false}
              onLoad={(e) => {
                const img = e.currentTarget;
                setAfterSize({
                  width: img.naturalWidth,
                  height: img.naturalHeight,
                });
              }}
              onError={() =>
                setLoadError("The enhanced image could not be loaded.")
              }
              style={layerStyle()}
            />
          </div>
        )}

        {/* divider */}
        {splitActive && container.width > 0 && (
          <div
            className="pointer-events-none absolute inset-y-0 z-10"
            style={{ left: `${dividerX}px` }}
          >
            <div className="absolute inset-y-0 -left-px w-0.5 bg-white/90 shadow-[0_0_6px_rgba(0,0,0,0.65)]" />
            <button
              type="button"
              aria-label="Comparison divider"
              role="slider"
              aria-orientation="vertical"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(divider * 100)}
              onPointerDown={(e) => {
                e.stopPropagation();
                const parent = containerRef.current;
                if (!parent) return;
                parent.focus();
                gesture.current = { kind: "divider" };
                parent.setPointerCapture(e.pointerId);
                pointers.current.set(e.pointerId, { x: 0, y: 0 });
              }}
              className={cn(
                "pointer-events-auto absolute top-1/2 left-1/2 flex size-8 -translate-x-1/2 -translate-y-1/2",
                "cursor-ew-resize items-center justify-center rounded-full",
                "border border-white/70 bg-black/55 text-white backdrop-blur-sm",
                "shadow-lg transition-transform hover:scale-105 active:scale-95",
              )}
            >
              <MoveHorizontal className="size-4" />
            </button>
          </div>
        )}

        {/* labels */}
        {ready && mode !== "after" && (
          <span className="pointer-events-none absolute bottom-3 left-3 z-10 rounded bg-black/55 px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-white/90 uppercase backdrop-blur-sm">
            {beforeLabel}
          </span>
        )}
        {ready && afterSrc && mode !== "before" && (
          <span className="pointer-events-none absolute right-3 bottom-3 z-10 rounded bg-black/55 px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-white/90 uppercase backdrop-blur-sm">
            {afterLabel}
          </span>
        )}

        {!ready && !loadError && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-fg-subtle">
            Loading image...
          </div>
        )}

        {loadError && (
          <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-xs text-danger">
            {loadError}
          </div>
        )}

        {overlay}
      </div>

      {/* hint bar */}
      <div className="flex items-center gap-3 border-t border-border bg-panel px-3 py-1 text-[10px] text-fg-subtle">
        <span className="inline-flex items-center gap-1">
          <Maximize2 className="size-3" /> scroll to zoom
        </span>
        <span>drag to pan</span>
        {splitActive && <span>drag the handle to compare</span>}
        <span className="ml-auto hidden sm:inline">
          0 fit &middot; 1 actual pixels &middot; arrows move divider
        </span>
      </div>
    </div>
  );
}
