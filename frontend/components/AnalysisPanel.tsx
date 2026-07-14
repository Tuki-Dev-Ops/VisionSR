"use client";

import { AlertTriangle, Loader2, ScanFace, Sparkles } from "lucide-react";
import { Meter } from "@/components/Meter";
import { Panel, PanelBody, PanelHeader, Row } from "@/components/ui/panel";
import { Tooltip } from "@/components/ui/tooltip";
import { useModels } from "@/hooks/useBackend";
import type { Analysis, ModelInfo } from "@/lib/types";
import { cn, formatPercent } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

function modelLabel(id: string | null, models: ModelInfo[] | undefined): string {
  if (!id) return "None";
  return models?.find((m) => m.id === id)?.name ?? id;
}

function confidenceTone(confidence: number): string {
  if (confidence >= 0.75) return "text-success";
  if (confidence >= 0.45) return "text-warn";
  return "text-danger";
}

export function AnalysisPanel() {
  const analysis = useAppStore((s) => s.analysis);
  const analyzing = useAppStore((s) => s.analyzing);
  const analysisError = useAppStore((s) => s.analysisError);
  const overrideId = useAppStore((s) => s.settings.modelId);
  const hasSource = useAppStore((s) => s.source !== null);
  const { data: models } = useModels();

  return (
    <Panel>
      <PanelHeader
        title="Analysis"
        actions={
          analyzing ? (
            <span className="flex items-center gap-1.5 text-[11px] text-fg-subtle">
              <Loader2 className="size-3 animate-spin" />
              Inspecting
            </span>
          ) : null
        }
      />
      <PanelBody>
        {!hasSource && (
          <p className="py-2 text-[12px] text-fg-subtle">
            Load an image to see what the engine detects.
          </p>
        )}

        {hasSource && analysisError && (
          <div className="flex items-start gap-2 rounded-md border border-danger/30 bg-danger-soft p-2 text-[11px] text-danger">
            <AlertTriangle className="mt-px size-3.5 shrink-0" />
            <span>
              Analysis unavailable. {analysisError} You can still run the
              upscale.
            </span>
          </div>
        )}

        {hasSource && !analysis && analyzing && (
          <div className="space-y-3">
            {[0, 1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-1 w-full animate-pulse rounded-full bg-panel-inset"
              />
            ))}
          </div>
        )}

        {analysis && <AnalysisBody analysis={analysis} models={models} overrideId={overrideId} />}
      </PanelBody>
    </Panel>
  );
}

function AnalysisBody({
  analysis,
  models,
  overrideId,
}: {
  analysis: Analysis;
  models: ModelInfo[] | undefined;
  overrideId: string | null;
}) {
  const faceCount = analysis.faces.length;

  return (
    <div className="space-y-3.5">
      {/* content type */}
      <div className="flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-panel-inset px-2 py-1 text-[12px] font-medium text-fg capitalize">
          {analysis.content_type.replace(/[_-]+/g, " ")}
        </span>
        <Tooltip label="How confident the classifier is about the content type.">
          <span
            className={cn(
              "tabular cursor-help text-[11px]",
              confidenceTone(analysis.content_confidence),
            )}
          >
            {formatPercent(analysis.content_confidence)} confidence
          </span>
        </Tooltip>
      </div>

      {/* geometry */}
      <div className="space-y-1.5">
        <Row
          label="Source"
          value={`${analysis.width.toLocaleString()} x ${analysis.height.toLocaleString()}`}
        />
        <Row
          label="Megapixels"
          value={`${analysis.megapixels.toFixed(2)} MP`}
        />
        <Row label="Alpha" value={analysis.has_alpha ? "Yes" : "No"} />
      </div>

      <div className="h-px bg-border" />

      {/* meters */}
      <div className="space-y-2.5">
        <Meter
          label="Quality"
          value={analysis.quality_score}
          polarity="higher-better"
          hint="Overall estimated quality of the source. Higher is better."
        />
        <Meter
          label="Noise"
          value={analysis.noise_level}
          polarity="higher-worse"
          hint="Estimated sensor/ISO noise. Higher means the denoiser has more to do."
        />
        <Meter
          label="Blur"
          value={analysis.blur_level}
          polarity="higher-worse"
          hint="Estimated defocus / motion blur. Higher means a softer source."
        />
        <Meter
          label="JPEG artifacts"
          value={analysis.compression_level}
          polarity="higher-worse"
          hint="Estimated block and ringing artifacts from lossy compression."
        />
      </div>

      <div className="h-px bg-border" />

      {/* faces */}
      <div className="flex items-center justify-between gap-2 text-[12px]">
        <span className="inline-flex items-center gap-1.5 text-fg-subtle">
          <ScanFace className="size-3.5" />
          Faces
        </span>
        <span className="tabular text-fg">
          {faceCount === 0
            ? "None detected"
            : `${faceCount} detected`}
        </span>
      </div>

      {/* model routing */}
      <div className="space-y-1.5 rounded-md border border-border bg-panel-inset p-2">
        <div className="flex items-center gap-1.5 text-[10px] font-medium tracking-wider text-fg-subtle uppercase">
          <Sparkles className="size-3 text-accent" />
          Auto-selected
        </div>
        <Row
          label="Model"
          value={modelLabel(analysis.recommended_model, models)}
        />
        {(faceCount > 0 || analysis.recommended_face_model) && (
          <Row
            label="Face model"
            value={modelLabel(analysis.recommended_face_model, models)}
          />
        )}
        {overrideId && overrideId !== analysis.recommended_model && (
          <p className="pt-1 text-[11px] text-warn">
            Overridden with {modelLabel(overrideId, models)}.
          </p>
        )}
      </div>
    </div>
  );
}
