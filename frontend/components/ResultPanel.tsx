"use client";

import { Download, Layers, Timer } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Panel, PanelBody, PanelHeader, Row } from "@/components/ui/panel";
import { useModels } from "@/hooks/useBackend";
import { resultHasAlpha } from "@/lib/models";
import { formatDuration, resultFilename } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

export function ResultPanel() {
  const job = useAppStore((s) => s.job);
  const source = useAppStore((s) => s.source);
  const { data: models } = useModels();

  if (!job || job.status !== "done" || !job.result) return null;

  const { result } = job;

  /* The settings the job ran with, not the ones currently in the panel — those stay
     live and the user may well have moved on. Using them here would let the download
     button offer PNG bytes under a ".jpg" name. */
  const { scale, outputFormat } = job.settings;
  const filename = resultFilename(source?.name ?? "image", scale, outputFormat);

  const transparent = resultHasAlpha(result, models);

  const modelNames = result.models_used.map(
    (id) => models?.find((m) => m.id === id)?.name ?? id,
  );

  return (
    <Panel>
      <PanelHeader
        title="Result"
        actions={
          <span className="inline-flex items-center gap-1 text-[11px] text-fg-subtle">
            <Timer className="size-3" />
            <span className="tabular">{formatDuration(result.duration_ms)}</span>
          </span>
        }
      />
      <PanelBody>
        <div className="space-y-1.5">
          <Row
            label="Output"
            value={`${result.width.toLocaleString()} x ${result.height.toLocaleString()}`}
          />
          <Row
            label="Megapixels"
            value={`${((result.width * result.height) / 1_000_000).toFixed(1)} MP`}
          />
          <Row label="Backend" value={result.backend} />
          <Row
            label="Tiles"
            value={
              <span className="inline-flex items-center gap-1">
                <Layers className="size-3 text-fg-subtle" />
                {result.tiles.toLocaleString()}
              </span>
            }
          />
          {transparent && (
            <Row
              label="Transparency"
              value={<span className="text-success">RGBA cutout</span>}
              title="The background was removed — this image has a real alpha channel."
            />
          )}
        </div>

        {/* Not a Row: that truncates its value to one right-aligned line, which was
            fine for the one or two models a plain upscale reports but turns a
            three-model pipeline into "Real-ESRGAN x4plus, GFP...". A chained pipeline
            is exactly what the user wants to read back, so give it the space. */}
        <div className="space-y-1.5">
          <span className="text-[12px] text-fg-subtle">Models</span>
          {modelNames.length > 0 ? (
            <ul className="flex flex-wrap gap-1">
              {modelNames.map((name, i) => (
                <li
                  key={`${result.models_used[i]}`}
                  className="rounded border border-border bg-panel-inset px-1.5 py-0.5 text-[11px] text-fg"
                >
                  {name}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[12px] text-fg">—</p>
          )}
        </div>

        {job.resultUrl && (
          <Button variant="primary" size="lg" className="w-full" asChild>
            <a href={job.resultUrl} download={filename}>
              <Download />
              Download {outputFormat.toUpperCase()}
            </a>
          </Button>
        )}
      </PanelBody>
    </Panel>
  );
}
