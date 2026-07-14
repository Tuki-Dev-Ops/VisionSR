"use client";

import { Tooltip } from "@/components/ui/tooltip";
import { clamp, cn } from "@/lib/utils";

interface MeterProps {
  label: string;
  /** 0..1 */
  value: number;
  /**
   * "higher-better" — quality: 1.0 is great.
   * "higher-worse"  — noise / blur / compression: 1.0 is bad.
   */
  polarity: "higher-better" | "higher-worse";
  hint?: string;
}

/** Severity 0 (good) .. 1 (bad), regardless of the metric's polarity. */
function severityOf(value: number, polarity: MeterProps["polarity"]): number {
  return polarity === "higher-worse" ? value : 1 - value;
}

function toneOf(severity: number): { bar: string; text: string } {
  if (severity < 0.34) return { bar: "bg-success", text: "text-success" };
  if (severity < 0.67) return { bar: "bg-warn", text: "text-warn" };
  return { bar: "bg-danger", text: "text-danger" };
}

export function Meter({ label, value, polarity, hint }: MeterProps) {
  const v = clamp(value, 0, 1);
  const tone = toneOf(severityOf(v, polarity));

  const bar = (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] text-fg-subtle">{label}</span>
        <span className={cn("tabular text-[11px] font-medium", tone.text)}>
          {v.toFixed(2)}
        </span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-panel-inset ring-1 ring-border ring-inset">
        <div
          className={cn("h-full rounded-full transition-[width]", tone.bar)}
          style={{ width: `${Math.max(v * 100, 1.5)}%` }}
          role="meter"
          aria-label={label}
          aria-valuenow={Number(v.toFixed(2))}
          aria-valuemin={0}
          aria-valuemax={1}
        />
      </div>
    </div>
  );

  if (!hint) return bar;
  return (
    <Tooltip label={hint}>
      <div className="cursor-help">{bar}</div>
    </Tooltip>
  );
}
