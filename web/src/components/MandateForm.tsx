/** Stating the mandate.
 *
 * The form is ordered the way an instruction is actually written: what must be
 * achieved, what rules bind the allocation, what the commodities sleeve is
 * made of, and only then the period and how the portfolio is held. Grouping
 * the inputs by the engine's module boundaries instead would be organising the
 * interface around how it was built.
 *
 * Every requirement is optional on its own but at least one is needed, which
 * the API enforces. A blank field means "no limit", so an empty form is a real
 * state rather than an incomplete one.
 */

import type { Meta } from "@/lib/api";
import {
  Button,
  Explainer,
  Field,
  Input,
  Panel,
  Select,
  Slider,
} from "@/components/ui/primitives";
import { PolicyLimits, emptyLimits, type Limits } from "@/components/PolicyLimits";

export interface FormState {
  start: string;
  end: string;
  targetReturn: string;
  maxVolatility: string;
  maxDrawdown: string;
  maxRecovery: string;
  goldWeight: number;
  limits: Limits;
  growthCap: string;
  rebalance: string;
  costBps: string;
  assets: string[];
}

const DEFAULT_ASSETS = [
  "equity",
  "fixed_income",
  "private_equity",
  "commodities",
  "cash",
];

export const INITIAL: FormState = {
  start: "",
  end: "",
  targetReturn: "6",
  maxVolatility: "10",
  maxDrawdown: "",
  maxRecovery: "",
  goldWeight: 0.5,
  limits: {
    ...emptyLimits(DEFAULT_ASSETS),
    cash: { floor: "5", cap: "" },
    private_equity: { floor: "", cap: "20" },
  },
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
  const selected =
    state.assets.length === 0 ? allocatable.map((a) => a.key) : state.assets;
  const inPlay = allocatable.filter((a) => selected.includes(a.key));

  const toggle = (key: string) => {
    const next = selected.includes(key)
      ? selected.filter((k) => k !== key)
      : [...selected, key];
    // Never allow an empty universe: there would be nothing to allocate across,
    // and the request would fail with a less obvious message than this.
    if (next.length > 0) set("assets", next);
  };

  const hasCommodities = selected.includes("commodities");

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
          <Field
            label="Drawdown, no worse than"
            hint="%, blank for none"
            explain={
              <Explainer title="drawdown">
                The deepest fall from a previous high, and the one number an
                investor actually lives through. Volatility treats a rise and a
                fall alike; drawdown counts only the fall, and only in the order
                it happened.
              </Explainer>
            }
          >
            <Input
              inputMode="decimal"
              value={state.maxDrawdown}
              onChange={(e) => set("maxDrawdown", e.target.value)}
              placeholder="25"
            />
          </Field>
          <Field
            label="Recovery, within"
            hint="months, blank for none"
            explain={
              <Explainer title="recovery">
                How long it took to climb back to the previous high after the
                worst fall. Two portfolios can share a 20% drawdown and be very
                different propositions if one is whole in eight months and the
                other takes forty.
              </Explainer>
            }
          >
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
        <PolicyLimits
          assets={inPlay}
          limits={state.limits}
          onChange={(limits) => set("limits", limits)}
          groupCap={state.growthCap}
          onGroupCapChange={(value) => set("growthCap", value)}
        />
      </Panel>

      {hasCommodities && (
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
          <dl className="mt-3 space-y-1.5 border-t border-line pt-3 text-xs text-muted">
            {meta.sleeve.components.map((component) => (
              <div key={component.key} className="flex gap-2">
                <dt className="w-20 shrink-0">{component.label}</dt>
                <dd>{component.proxy}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      )}

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
          <Field
            label="Rebalanced"
            explain={
              <Explainer title="rebalancing">
                A portfolio drifts: whatever rises grows into a larger share, so
                a 60/40 split can end up 75/25 without anyone deciding it.
                Rebalancing sells what rose and buys what fell to restore the
                target — controlling risk, at the cost of trading. Never
                rebalancing is cheapest and ends up holding whatever won.
              </Explainer>
            }
          >
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
          <Field
            label="Trading cost"
            hint="bps per trade"
            explain={
              <Explainer title="trading cost">
                What each rebalance costs, in hundredths of a percent of the
                amount traded. It decides whether frequent rebalancing is worth
                it: monthly trades roughly twelve times as often as annual, so a
                schedule that looks better before costs can lose after them.
                10bps is a reasonable institutional assumption.
              </Explainer>
            }
          >
            <Input
              inputMode="decimal"
              value={state.costBps}
              onChange={(e) => set("costBps", e.target.value)}
            />
          </Field>
        </div>

        <div className="mt-4 border-t border-line pt-4">
          <span className="eyebrow mb-2 block">asset classes</span>
          <div className="flex flex-wrap gap-2">
            {allocatable.map((asset) => {
              const on = selected.includes(asset.key);
              return (
                <button
                  key={asset.key}
                  type="button"
                  onClick={() => toggle(asset.key)}
                  title={`${asset.proxy} — ${asset.caveat}`}
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
        </div>
      </Panel>

      <Button type="submit" disabled={pending} className="w-full">
        {pending ? "Solving…" : "Find allocations"}
      </Button>
    </form>
  );
}
