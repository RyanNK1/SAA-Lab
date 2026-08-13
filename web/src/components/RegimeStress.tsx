/** How candidate allocations fared through each regime.
 *
 * This view deliberately does not optimise per period. A period-by-period
 * optimum is a corner solution -- everything into fixed income through the
 * crisis, everything into gold through the inflation shock -- and no committee
 * would hold any of them, so what they endured says nothing useful. Knowing
 * that a hindsight-perfect allocation did well in the year it was chosen for
 * is not information.
 *
 * What is worth knowing is how an allocation someone might *actually own* held
 * up when conditions changed. So the rows here come from the mandate: real
 * candidates that met a real requirement, then measured through every regime
 * they were not chosen for.
 *
 * The colour scale is fixed rather than relative to the data, so a cell means
 * the same thing between one comparison and the next.
 */

import { useState } from "react";
import type { TrackResult, TrackedAllocation } from "@/lib/api";
import { assetLabel, num, pct } from "@/lib/format";
import { Explainer, Panel, Select } from "@/components/ui/primitives";

type Measure = "realised_return" | "max_drawdown" | "sharpe" | "volatility";

const MEASURES: { key: Measure; label: string; format: (v: number) => string }[] = [
  { key: "realised_return", label: "return", format: (v) => pct(v, 1) },
  { key: "max_drawdown", label: "drawdown", format: (v) => pct(v, 1) },
  { key: "sharpe", label: "Sharpe", format: (v) => num(v, 2) },
  { key: "volatility", label: "volatility", format: (v) => pct(v, 1) },
];

/** Scales chosen so the midpoint of each is a genuinely neutral outcome: zero
 *  for return and Sharpe, and for drawdown the point where a fall stops being
 *  ordinary. Using the data's own range instead would make a set of uniformly
 *  bad results look varied. */
const SCALE: Record<Measure, { neutral: number; span: number; inverted: boolean }> = {
  realised_return: { neutral: 0, span: 0.15, inverted: false },
  max_drawdown: { neutral: -0.1, span: 0.25, inverted: false },
  sharpe: { neutral: 0, span: 1.5, inverted: false },
  volatility: { neutral: 0.1, span: 0.15, inverted: true },
};

function shade(value: number, measure: Measure) {
  const { neutral, span, inverted } = SCALE[measure];
  const signed = ((value - neutral) / span) * (inverted ? -1 : 1);
  const weight = Math.min(Math.abs(signed), 1);
  const colour = signed >= 0 ? "var(--color-primary)" : "var(--color-brick)";

  return {
    background: `color-mix(in srgb, ${colour} ${weight * 42}%, transparent)`,
    color: weight > 0.6 ? "var(--color-surface)" : "var(--color-ink)",
  };
}

function Weights({ allocation }: { allocation: TrackedAllocation }) {
  const held = Object.entries(allocation.weights)
    .filter(([, weight]) => weight > 0.005)
    .sort((a, b) => b[1] - a[1]);

  return (
    <span className="tabular text-[0.6875rem] text-muted">
      {held.map(([asset, weight]) => `${assetLabel(asset).split(" ")[0]} ${pct(weight, 0)}`).join(" · ")}
    </span>
  );
}

export function RegimeStress({
  result,
  selectedPeriods,
  onPeriodsChange,
  allPeriods,
}: {
  result: TrackResult;
  selectedPeriods: string[];
  onPeriodsChange: (next: string[]) => void;
  allPeriods: { label: string; start: string; end: string; note: string }[];
}) {
  const [measure, setMeasure] = useState<Measure>("realised_return");
  const active = MEASURES.find((m) => m.key === measure)!;
  const periods = result.periods.map((p) => p.label);

  const togglePeriod = (label: string) => {
    const next = selectedPeriods.includes(label)
      ? selectedPeriods.filter((l) => l !== label)
      : [...selectedPeriods, label];
    if (next.length > 0) onPeriodsChange(next);
  };

  const struggled = result.allocations.filter((a) => a.negative_periods > 0);

  return (
    <div className="space-y-5">
      <Panel title="regimes">
        <div className="flex flex-wrap gap-2">
          {allPeriods.map((period) => {
            const on = selectedPeriods.includes(period.label);
            return (
              <span key={period.label} className="inline-flex items-center">
                <button
                  type="button"
                  onClick={() => togglePeriod(period.label)}
                  aria-pressed={on}
                  className={
                    on
                      ? "border border-primary bg-primary-soft px-2.5 py-1 text-xs text-primary"
                      : "border border-line px-2.5 py-1 text-xs text-muted hover:border-line-strong"
                  }
                >
                  {period.label}
                </button>
                <Explainer title={period.label}>
                  <span className="tabular mb-1.5 block text-[0.6875rem] text-ink">
                    {period.start} to {period.end}
                  </span>
                  {period.note}
                </Explainer>
              </span>
            );
          })}
        </div>
        <p className="mt-3 text-xs text-muted">
          Windows chosen for how the world behaved rather than round numbers.
          Deselect any that are not the question you are asking.
        </p>
      </Panel>

      <Panel
        title={`${active.label} by regime`}
        aside={
          <Select
            value={measure}
            onChange={(e) => setMeasure(e.target.value as Measure)}
            className="w-auto py-1 text-xs"
          >
            {MEASURES.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </Select>
        }
      >
        <div className="-mx-5 overflow-x-auto px-5">
          <table className="w-full min-w-[46rem] border-collapse text-xs">
            <thead>
              <tr>
                <th className="eyebrow py-2 pr-3 text-left font-medium">
                  allocation
                </th>
                {periods.map((label) => (
                  <th key={label} className="eyebrow px-1 py-2 text-center font-medium">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.allocations.map((allocation) => (
                <tr key={allocation.label}>
                  <td className="py-1 pr-3 whitespace-nowrap">
                    <div className="text-[0.8125rem] text-ink">{allocation.label}</div>
                    <Weights allocation={allocation} />
                  </td>
                  {allocation.by_period.map((entry) => {
                    const value = entry[measure];
                    return (
                      <td key={entry.period} className="p-0.5">
                        <div
                          className="tabular py-2 text-center"
                          style={shade(value, measure)}
                          title={`${allocation.label} · ${entry.period} · ${entry.months} months`}
                        >
                          {active.format(value)}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-muted">
          Every allocation met the mandate over the whole sample. These are the
          same allocations measured in conditions they were not chosen for.
        </p>
      </Panel>

      <Panel title="prosperity and struggle">
        <div className="space-y-0">
          {result.allocations.map((allocation) => (
            <div
              key={allocation.label}
              className="grid grid-cols-[8rem_1fr_1fr_5.5rem] items-baseline gap-4 border-t border-line py-3 first:border-t-0 first:pt-0"
            >
              <span className="text-[0.8125rem] text-ink">{allocation.label}</span>
              <span className="text-xs text-muted">
                strongest in{" "}
                <span className="text-ink">{allocation.best_period ?? "--"}</span>
              </span>
              <span className="text-xs text-muted">
                worst in{" "}
                <span className="text-brick">{allocation.worst_period ?? "--"}</span>
              </span>
              <span className="tabular text-right text-xs text-muted">
                {allocation.negative_periods} of {periods.length} down
              </span>
            </div>
          ))}
        </div>

        <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-muted">
          {struggled.length === 0
            ? "None of these lost money in any regime shown — unusual, and worth checking against a longer window before trusting it."
            : `${struggled.length} of ${result.allocations.length} lost money in at least one regime. An allocation that met the mandate over twenty years still had periods it would have been painful to hold, and those are the periods people sell in.`}
        </p>
      </Panel>
    </div>
  );
}
