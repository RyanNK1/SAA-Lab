/** Stating the mandate.
 *
 * The form is ordered the way an instruction is actually written: what must be
 * achieved, what may not be exceeded, what rules bind the allocation, and only
 * then how it is held. Grouping the inputs by the engine's module boundaries
 * instead would be organising the interface around how it was built.
 *
 * Every requirement is optional individually but at least one is needed, which
 * the API enforces. Leaving a field blank means "no limit", so an empty form is
 * a real state rather than an incomplete one.
 */

import type { Meta } from "@/lib/api";
import { Button, Field, Input, Panel, Select, Slider } from "@/components/ui/primitives";

export interface FormState {
  start: string;
  end: string;
  targetReturn: string;
  maxVolatility: string;
  maxDrawdown: string;
  maxRecovery: string;
  goldWeight: number;
  cashFloor: string;
  peCap: string;
  growthCap: string;
  rebalance: string;
  costBps: string;
  assets: string[];
}

export const INITIAL: FormState = {
  start: "",
  end: "",
  targetReturn: "6",
  maxVolatility: "10",
  maxDrawdown: "",
  maxRecovery: "",
  goldWeight: 0.5,
  cashFloor: "5",
  peCap: "20",
  growthCap: "60",
  rebalance: "monthly",
  costBps: "0",
  assets: [],
};

/** A percentage typed by a person, as the fraction the API expects. Blank
 *  means the requirement is not set, which is different from zero. */
export const asFraction = (value: string): number | null => {
  const trimmed = value.trim();
  if (trimmed === "") return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed / 100 : null;
};

export function MandateForm({
  meta,
  state,
  onChange,
  onSubmit,
  pending,
}: {
  meta: Meta;
  state: FormState;
  onChange: (next: FormState) => void;
  onSubmit: () => void;
  pending: boolean;
}) {
  const set = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    onChange({ ...state, [key]: value });

  const allocatable = meta.assets.filter((a) => a.allocatable);
  const selected = state.assets.length === 0 ? allocatable.map((a) => a.key) : state.assets;

  const toggle = (key: string) => {
    const next = selected.includes(key)
      ? selected.filter((k) => k !== key)
      : [...selected, key];
    // Never allow an empty universe: there would be nothing to allocate across
    // and the request would fail with a less obvious message.
    if (next.length > 0) set("assets", next);
  };

  return (
    <form
      className="space-y-5"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <Panel title="must achieve">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Return, at least" hint="% a year">
            <Input
              inputMode="decimal"
              value={state.targetReturn}
              onChange={(e) => set("targetReturn", e.target.value)}
              placeholder="6"
            />
          </Field>
          <Field label="Volatility, at most" hint="% a year">
            <Input
              inputMode="decimal"
              value={state.maxVolatility}
              onChange={(e) => set("maxVolatility", e.target.value)}
              placeholder="10"
            />
          </Field>
          <Field label="Drawdown, no worse than" hint="%, leave blank for none">
            <Input
              inputMode="decimal"
              value={state.maxDrawdown}
              onChange={(e) => set("maxDrawdown", e.target.value)}
              placeholder="25"
            />
          </Field>
          <Field label="Recovery, within" hint="months, blank for none">
            <Input
              inputMode="numeric"
              value={state.maxRecovery}
              onChange={(e) => set("maxRecovery", e.target.value)}
              placeholder="24"
            />
          </Field>
        </div>
      </Panel>

      <Panel title="policy limits">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Cash, at least" hint="%">
            <Input
              inputMode="decimal"
              value={state.cashFloor}
              onChange={(e) => set("cashFloor", e.target.value)}
            />
          </Field>
          <Field label="Private equity, at most" hint="%">
            <Input
              inputMode="decimal"
              value={state.peCap}
              onChange={(e) => set("peCap", e.target.value)}
            />
          </Field>
          <Field
            label="Growth assets, at most"
            hint="equity + private equity, %"
          >
            <Input
              inputMode="decimal"
              value={state.growthCap}
              onChange={(e) => set("growthCap", e.target.value)}
            />
          </Field>
        </div>
      </Panel>

      <Panel title="the commodities sleeve">
        <Slider
          value={state.goldWeight}
          onChange={(value) => set("goldWeight", value)}
          leftLabel="gold"
          rightLabel="ex-gold"
        />
        <p className="mt-3 text-xs leading-relaxed text-muted">
          One bucket, split between gold and everything else. The two behave
          very differently — gold rose through the 2008 crisis while broad
          commodities fell with equities — so this changes what the sleeve is,
          not just what it holds.
        </p>
      </Panel>

      <Panel title="period and holding">
        <div className="grid grid-cols-2 gap-4">
          <Field label="From" hint={`data starts ${meta.coverage.start}`}>
            <Input
              type="date"
              value={state.start}
              min={meta.coverage.start}
              max={meta.coverage.end}
              onChange={(e) => set("start", e.target.value)}
            />
          </Field>
          <Field label="To" hint={`ends ${meta.coverage.end}`}>
            <Input
              type="date"
              value={state.end}
              min={meta.coverage.start}
              max={meta.coverage.end}
              onChange={(e) => set("end", e.target.value)}
            />
          </Field>
          <Field label="Rebalanced">
            <Select
              value={state.rebalance}
              onChange={(e) => set("rebalance", e.target.value)}
            >
              {meta.rebalance_schedules.map((schedule) => (
                <option key={schedule} value={schedule}>
                  {schedule}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Trading cost" hint="bps per trade">
            <Input
              inputMode="decimal"
              value={state.costBps}
              onChange={(e) => set("costBps", e.target.value)}
            />
          </Field>
        </div>

        <div className="mt-4 flex flex-wrap gap-2 border-t border-line pt-4">
          {allocatable.map((asset) => {
            const on = selected.includes(asset.key);
            return (
              <button
                key={asset.key}
                type="button"
                onClick={() => toggle(asset.key)}
                title={asset.caveat}
                aria-pressed={on}
                className={
                  on
                    ? "border border-primary bg-primary-soft px-2.5 py-1 text-xs text-primary"
                    : "border border-line px-2.5 py-1 text-xs text-muted hover:border-line-strong"
                }
              >
                {asset.label}
              </button>
            );
          })}
        </div>
      </Panel>

      <Button type="submit" disabled={pending} className="w-full">
        {pending ? "Solving…" : "Find allocations"}
      </Button>
    </form>
  );
}
