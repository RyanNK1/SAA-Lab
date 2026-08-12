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
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[0.8125rem] font-medium text-ink">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </label>
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
