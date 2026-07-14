"use client";

import * as React from "react";
import { Loader2, RotateCcw, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Segmented, type SegmentedOption } from "@/components/ui/segmented";
import { Slider } from "@/components/ui/slider";
import { Tooltip } from "@/components/ui/tooltip";
import { useModels } from "@/hooks/useBackend";
import { useRunJob } from "@/hooks/useRunJob";
import {
  backgroundRemovalModels,
  superResolutionModels,
} from "@/lib/models";
import {
  MAX_BACKGROUND_FEATHER,
  OUTPUT_FORMATS,
  SCALES,
  formatSupportsAlpha,
  type FaceRestoreMode,
  type ModelInfo,
  type OutputFormat,
  type Scale,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { DEFAULT_SETTINGS, selectIsRunning, useAppStore } from "@/store/appStore";

const AUTO = "__auto__";

/** Segmented is typed on string|number, so the boolean toggle rides on this pair. */
type Toggle = "off" | "on";

const SCALE_OPTIONS: readonly SegmentedOption<Scale>[] = SCALES.map((s) => ({
  value: s,
  label: `${s}x`,
}));

const FACE_OPTIONS: readonly SegmentedOption<FaceRestoreMode>[] = [
  { value: "auto", label: "Auto", title: "Restore faces only if the engine detects them" },
  { value: "on", label: "On", title: "Always run face restoration" },
  { value: "off", label: "Off", title: "Never run face restoration" },
];

const BACKGROUND_OPTIONS: readonly SegmentedOption<Toggle>[] = [
  { value: "off", label: "Off", title: "Keep the background" },
  {
    value: "on",
    label: "On",
    title: "Cut the subject out — the result is a transparent PNG/WebP",
  },
];

const FORMAT_LABELS: Record<OutputFormat, string> = {
  png: "PNG (lossless)",
  jpeg: "JPEG",
  webp: "WebP",
};

function Field({
  label,
  hint,
  value,
  children,
}: {
  label: string;
  hint?: string;
  value?: string;
  children: React.ReactNode;
}) {
  const heading = (
    <div className="flex items-baseline justify-between gap-2">
      <span
        className={cn(
          "text-[11px] font-medium tracking-wide text-fg-muted",
          hint && "cursor-help decoration-dotted underline-offset-2 hover:underline",
        )}
      >
        {label}
      </span>
      {value && <span className="tabular text-[11px] text-fg-subtle">{value}</span>}
    </div>
  );

  return (
    <div className="space-y-1.5">
      {hint ? <Tooltip label={hint}>{heading}</Tooltip> : heading}
      {children}
    </div>
  );
}

/**
 * Group models by their `task`, so the dropdown stays honest about what's what.
 *
 * Allow-list, not a deny-list. This dropdown chooses the *upscaler*, and every other
 * task in the registry is driven by its own control — face restoration by the face
 * segmented control, segmentation by the background one. Filtering by "not a face
 * model" used to be equivalent; it stopped being so the moment background_removal
 * models joined the same list, and it would have quietly offered IS-Net as an
 * upscaler. Naming what belongs here means the next task added to the registry is
 * excluded by default rather than leaking into the UI.
 */
function groupModels(models: ModelInfo[]): Map<string, ModelInfo[]> {
  const groups = new Map<string, ModelInfo[]>();
  for (const model of superResolutionModels(models)) {
    const key = model.task || "models";
    const list = groups.get(key) ?? [];
    list.push(model);
    groups.set(key, list);
  }
  return groups;
}

export function ControlsPanel() {
  const settings = useAppStore((s) => s.settings);
  const setSetting = useAppStore((s) => s.setSetting);
  const resetSettings = useAppStore((s) => s.resetSettings);
  const hasSource = useAppStore((s) => s.source !== null);
  const running = useAppStore(selectIsRunning);
  const { data: models, isError: modelsError } = useModels();
  const { run, cancel, submitting } = useRunJob();

  const busy = running || submitting;
  const locked = busy;

  const selected = models?.find((m) => m.id === settings.modelId) ?? null;
  const groups = React.useMemo(
    () => groupModels(models ?? []),
    [models],
  );
  const bgModels = React.useMemo(
    () => backgroundRemovalModels(models ?? []),
    [models],
  );

  const cutout = settings.removeBackground;

  const isDefault =
    settings.scale === DEFAULT_SETTINGS.scale &&
    settings.modelId === DEFAULT_SETTINGS.modelId &&
    settings.faceRestore === DEFAULT_SETTINGS.faceRestore &&
    settings.faceRestoreWeight === DEFAULT_SETTINGS.faceRestoreWeight &&
    settings.sharpen === DEFAULT_SETTINGS.sharpen &&
    settings.outputFormat === DEFAULT_SETTINGS.outputFormat &&
    settings.removeBackground === DEFAULT_SETTINGS.removeBackground &&
    settings.backgroundModelId === DEFAULT_SETTINGS.backgroundModelId &&
    settings.backgroundFeather === DEFAULT_SETTINGS.backgroundFeather;

  return (
    <Panel>
      <PanelHeader
        title="Settings"
        actions={
          <Tooltip label="Reset to defaults">
            <Button
              variant="ghost"
              size="icon"
              onClick={resetSettings}
              disabled={locked || isDefault}
              aria-label="Reset settings"
            >
              <RotateCcw />
            </Button>
          </Tooltip>
        }
      />
      <PanelBody className="space-y-4">
        <Field label="Scale">
          <Segmented
            value={settings.scale}
            options={SCALE_OPTIONS}
            onChange={(v) => setSetting("scale", v)}
            disabled={locked}
            ariaLabel="Upscale factor"
          />
        </Field>

        <Field
          label="Model"
          hint="Auto uses the model the analyzer recommends for this image."
        >
          <Select
            value={settings.modelId ?? AUTO}
            onValueChange={(value) =>
              setSetting("modelId", value === AUTO ? null : value)
            }
            disabled={locked}
          >
            <SelectTrigger aria-label="Model">
              <SelectValue placeholder="Auto" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={AUTO}>Auto (recommended)</SelectItem>
              {[...groups.entries()].map(([task, list]) => (
                <SelectGroup key={task}>
                  <SelectLabel>{task.replace(/[_-]+/g, " ")}</SelectLabel>
                  {list.map((model) => (
                    <SelectItem
                      key={model.id}
                      value={model.id}
                      disabled={!model.installed}
                    >
                      {model.name}
                      {model.installed ? "" : " (not installed)"}
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
          {modelsError && (
            <p className="text-[11px] text-danger">
              Could not load the model list. Auto still works.
            </p>
          )}
          {selected && (
            <p className="text-[11px] leading-relaxed text-fg-subtle">
              {selected.architecture} &middot; native {selected.scale}x
              {selected.scale !== settings.scale && (
                <span className="text-warn"> (requested {settings.scale}x)</span>
              )}
              {selected.description ? ` — ${selected.description}` : ""}
            </p>
          )}
        </Field>

        <div className="h-px bg-border" />

        <Field
          label="Face restore"
          hint="Runs a dedicated face model on detected faces. Auto only fires when faces are found."
        >
          <Segmented
            value={settings.faceRestore}
            options={FACE_OPTIONS}
            onChange={(v) => setSetting("faceRestore", v)}
            disabled={locked}
            ariaLabel="Face restoration mode"
          />
        </Field>

        <Field
          label="Face strength"
          value={settings.faceRestoreWeight.toFixed(2)}
          hint="How strongly the restored face is blended back in. 0 keeps the original face."
        >
          <Slider
            value={[settings.faceRestoreWeight]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={([v]) => setSetting("faceRestoreWeight", v ?? 0)}
            disabled={locked || settings.faceRestore === "off"}
            aria-label="Face restoration weight"
          />
        </Field>

        <div className="h-px bg-border" />

        <Field
          label="Remove background"
          hint="Runs a segmentation model and cuts the subject out. The result is RGBA with real transparency, so it can only be saved as PNG or WebP."
        >
          <Segmented
            value={cutout ? "on" : "off"}
            options={BACKGROUND_OPTIONS}
            onChange={(v) => setSetting("removeBackground", v === "on")}
            disabled={locked}
            ariaLabel="Remove background"
          />
        </Field>

        {cutout && (
          <div className="animate-fade-in space-y-4">
            <Field
              label="Background model"
              hint="Auto lets the engine pick. IS-Net is markedly better on hair and fur; U^2-Net is much smaller and faster."
            >
              <Select
                value={settings.backgroundModelId ?? AUTO}
                onValueChange={(value) =>
                  setSetting("backgroundModelId", value === AUTO ? null : value)
                }
                disabled={locked}
              >
                <SelectTrigger aria-label="Background model">
                  <SelectValue placeholder="Auto" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={AUTO}>Auto (recommended)</SelectItem>
                  {bgModels.map((model) => (
                    <SelectItem
                      key={model.id}
                      value={model.id}
                      disabled={!model.installed}
                    >
                      {model.name}
                      {model.installed ? "" : " (not installed)"}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field
              label="Edge feather"
              value={`${settings.backgroundFeather} px`}
              hint="Extra blur on the alpha edge. 0 keeps the model's own softness, which is usually right — reach for this only when the cutout edge is visibly hard against its new background."
            >
              <Slider
                value={[settings.backgroundFeather]}
                min={0}
                max={MAX_BACKGROUND_FEATHER}
                step={1}
                onValueChange={([v]) => setSetting("backgroundFeather", v ?? 0)}
                disabled={locked}
                aria-label="Edge feather"
              />
            </Field>
          </div>
        )}

        <div className="h-px bg-border" />

        <Field
          label="Sharpen"
          value={settings.sharpen.toFixed(2)}
          hint="Post-process unsharp amount applied after upscaling. 0 disables it."
        >
          <Slider
            value={[settings.sharpen]}
            min={0}
            max={1.5}
            step={0.05}
            onValueChange={([v]) => setSetting("sharpen", v ?? 0)}
            disabled={locked}
            aria-label="Sharpen amount"
          />
        </Field>

        <Field
          label="Output format"
          hint={
            cutout
              ? "JPEG cannot store an alpha channel: saving a cutout as JPEG would flatten it onto a white background without warning. Only PNG and WebP are offered while background removal is on."
              : undefined
          }
        >
          <Select
            value={settings.outputFormat}
            onValueChange={(value) =>
              setSetting("outputFormat", value as OutputFormat)
            }
            disabled={locked}
          >
            <SelectTrigger aria-label="Output format">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {OUTPUT_FORMATS.map((format) => {
                // A cutout is RGBA. JPEG has no alpha channel, so it cannot be an
                // option here — offering it would hand back a silently flattened,
                // white-backed image that looks like the feature simply failed.
                const unavailable = cutout && !formatSupportsAlpha(format);
                return (
                  <SelectItem
                    key={format}
                    value={format}
                    disabled={unavailable}
                  >
                    {FORMAT_LABELS[format]}
                    {unavailable ? " — no transparency" : ""}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
          {cutout && (
            <p className="text-[11px] leading-relaxed text-fg-subtle">
              Transparent result — PNG or WebP only.{" "}
              <span className="text-warn">JPEG has no alpha channel.</span>
            </p>
          )}
        </Field>

        <div className="flex gap-2 pt-1">
          <Button
            variant="primary"
            size="lg"
            className="flex-1"
            disabled={!hasSource || busy}
            onClick={run}
          >
            {busy ? (
              <>
                <Loader2 className="animate-spin" />
                Running
              </>
            ) : (
              <>
                <Sparkles />
                Upscale {settings.scale}x
              </>
            )}
          </Button>
          {busy && (
            <Tooltip label="Cancel this job">
              <Button
                variant="danger"
                size="lg"
                onClick={cancel}
                aria-label="Cancel job"
              >
                <X />
              </Button>
            </Tooltip>
          )}
        </div>
      </PanelBody>
    </Panel>
  );
}
