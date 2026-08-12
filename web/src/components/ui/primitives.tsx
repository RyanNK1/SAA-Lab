/** Interface primitives.
 *
 * Written here rather than pulled from a component library, so the visual
 * decisions live in this repository and can be changed without fighting a
 * dependency's defaults.
 *
 * Everything is square-cornered and hairline-ruled. The subject is
 * institutional allocation -- committee papers and terminal output -- and
 * rounded, shadowed cards would be borrowing a consumer-product vocabulary
 * that does not belong to it.
 */

import { useState } from "react";
import { cn } from "@/lib/utils";
import type { ReactNode, InputHTMLAttributes, SelectHTMLAttributes } from "react";

export function Panel({
  title,
  aside,
  children,
  className,
}: {
  title?: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("border border-line bg-surface", className)}>
      {title && (
        <header className="flex items-baseline justify-between gap-4 border-b border-line px-5 py-3">
          <h2 className="eyebrow">{title}</h2>
          {aside}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Field({
  label,
  hint,
  explain,
  children,
}: {
  label: string;
  hint?: string;
  explain?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="block">
      <span className="mb-1.5 flex items-center text-[0.8125rem] font-medium text-ink">
        {label}
        {explain}
      </span>
      {children}
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </div>
  );
}

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "tabular w-full border border-line bg-ground px-3 py-2 text-sm",
        "transition-colors placeholder:text-line-strong",
        "focus:border-primary focus:bg-surface focus:outline-none",
        className,
      )}
    />
  );
}

export function Select({
  className,
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(
        "w-full appearance-none border border-line bg-ground px-3 py-2 text-sm",
        "transition-colors focus:border-primary focus:bg-surface focus:outline-none",
        className,
      )}
    >
      {children}
    </select>
  );
}

export function Button({
  variant = "solid",
  className,
  children,
  ...props
}: InputHTMLAttributes<HTMLButtonElement> & {
  variant?: "solid" | "quiet";
  type?: "button" | "submit";
}) {
  return (
    <button
      {...props}
      className={cn(
        "px-4 py-2.5 text-sm font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-45",
        variant === "solid"
          ? "bg-primary text-surface hover:bg-ink"
          : "border border-line-strong bg-transparent text-ink hover:border-ink",
        className,
      )}
    >
      {children}
    </button>
  );
}

/** A labelled slider. Used for the commodities sleeve, where the value is a
 *  composition rather than a magnitude, so both ends are named. */
export function Slider({
  value,
  onChange,
  leftLabel,
  rightLabel,
}: {
  value: number;
  onChange: (value: number) => void;
  leftLabel: string;
  rightLabel: string;
}) {
  return (
    <div>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(value * 100)}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="h-1 w-full cursor-pointer appearance-none bg-line accent-brass"
        aria-label={`${leftLabel} share of the commodities sleeve`}
      />
      <div className="mt-1.5 flex justify-between text-xs text-muted">
        <span className="tabular">
          {Math.round(value * 100)}% {leftLabel}
        </span>
        <span className="tabular">
          {Math.round((1 - value) * 100)}% {rightLabel}
        </span>
      </div>
    </div>
  );
}

export function Stat({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "brick";
}) {
  return (
    <div>
      <div className="eyebrow mb-1">{label}</div>
      <div
        className={cn(
          "tabular text-xl",
          tone === "brick" ? "text-brick" : "text-ink",
        )}
      >
        {value}
      </div>
    </div>
  );
}

/** An explanation attached to a control that needs one.
 *
 * Rebalancing and trading cost both change the answer materially while meaning
 * nothing to someone who has not met them before, and a field whose effect is
 * invisible is a field people leave alone. The note opens on click rather than
 * hover so it works on a touchscreen and can be read at leisure.
 */
export function Explainer({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label={`What ${title} means`}
        className={cn(
          "ml-1.5 inline-flex h-4 w-4 items-center justify-center rounded-full border text-[0.625rem] leading-none transition-colors",
          open
            ? "border-primary bg-primary text-surface"
            : "border-line-strong text-muted hover:border-ink hover:text-ink",
        )}
      >
        i
      </button>

      {open && (
        <>
          <span
            className="fixed inset-0 z-10"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <span className="absolute left-0 top-6 z-20 block w-72 border border-line-strong bg-surface p-3 text-xs leading-relaxed font-normal text-muted shadow-[0_2px_12px_rgba(19,26,24,0.10)]">
            {children}
          </span>
        </>
      )}
    </span>
  );
}
