"use client";

import { cn } from "@/lib/utils";

export interface SegmentedOption<T extends string | number> {
  value: T;
  label: string;
  title?: string;
}

interface SegmentedProps<T extends string | number> {
  value: T;
  options: readonly SegmentedOption<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
  size?: "sm" | "md";
  className?: string;
  ariaLabel?: string;
}

/** Small radio-group-style toggle. Typed on the option union, not on string. */
export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  disabled = false,
  size = "md",
  className,
  ariaLabel,
}: SegmentedProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn(
        "grid w-full gap-0.5 rounded-md border border-border bg-panel-inset p-0.5",
        disabled && "pointer-events-none opacity-40",
        className,
      )}
      style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={String(option.value)}
            type="button"
            role="radio"
            aria-checked={selected}
            title={option.title}
            disabled={disabled}
            onClick={() => onChange(option.value)}
            className={cn(
              "rounded-[5px] font-medium transition-colors",
              size === "sm" ? "h-6 text-[11px]" : "h-7 text-xs",
              selected
                ? "bg-panel-raised text-fg shadow-sm ring-1 ring-border-strong"
                : "text-fg-subtle hover:text-fg",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
