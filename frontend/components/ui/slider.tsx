"use client";

import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { cn } from "@/lib/utils";

/**
 * `aria-label` is deliberately pulled off the props and handed to the Thumb.
 *
 * Radix puts role="slider" (and aria-valuenow) on the thumb, not the root — the root
 * is an unlabelled generic span. Spreading the caller's aria-label onto the root, as
 * this used to, therefore left the actual slider with no accessible name: a screen
 * reader announced "slider, 0.5" with no hint of what it controlled, for every slider
 * in the app. The label belongs on the element that carries the role.
 */
export const Slider = React.forwardRef<
  React.ComponentRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, "aria-label": ariaLabel, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex w-full touch-none items-center select-none",
      "data-[disabled]:opacity-40",
      className,
    )}
    {...props}
  >
    <SliderPrimitive.Track className="relative h-1 w-full grow overflow-hidden rounded-full bg-panel-inset ring-1 ring-border ring-inset">
      <SliderPrimitive.Range className="absolute h-full bg-accent" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb
      aria-label={ariaLabel}
      className={cn(
        "block size-3.5 rounded-full border-2 border-accent bg-panel-raised shadow",
        "transition-transform hover:scale-110 focus-visible:scale-110",
        "disabled:pointer-events-none",
      )}
    />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;
