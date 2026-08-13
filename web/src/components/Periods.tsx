/** How the answer changes between regimes.
 *
 * One period is a lookup. Several side by side is an argument, and this view
 * exists to make it: the best allocation for the 2008 crisis looks nothing
 * like the best one for the zero-rate decade, and an interface that only ever
 * showed a single window would let a user mistake one regime's answer for a
 * general one.
 *
 * The matrix is the centrepiece. Every row is an allocation chosen for one
 * period; every column is a period it was then measured in. The diagonal wins
 * by construction -- it was picked knowing what happened -- so what matters is
 * reading across a row to see whether it survived anywhere else. Usually it
 * did not.
 */

import type { PeriodsResult } from "@/lib/api";
import { assetLabel, num, pct } from "@/lib/format";
import { Panel, Stat } from "@/components/ui/primitives";

/** Sharpe ratios shaded from brick through neutral to petrol. The scale is
 *  fixed rather than relative to the data, so colour means the same thing
 *  between one comparison and the next. */
function shade(value: number | null): { background: string; color: string } {
  if (value === null || !Number.isFinite(value)) {
    return { background: "transparent", color: "var(--color-line-strong)" };
  }
  if (value < 0) {
    const weight = Math.min(Math.abs(value) / 1.5, 1);
    return {
      background: `color-mix(in srgb, var(--color-brick) ${weight * 45}%, transparent)`,
      color: weight > 0.55 ? "var(--color-surface)" : "var(--color-ink)",
    };
  }
  const weight = Math.min(value / 2, 1);
  return {
    background: `color-mix(in srgb, var(--color-primary) ${weight * 45}%, transparent)`,
    color: weight > 0.55 ? "var(--color-surface)" : "var(--color-ink)",
  };
}

function Matrix({ data }: { data: PeriodsResult["cross_period"] }) {
  return (
    <div className="-mx-5 overflow-x-auto px-5">
      <table className="w-full min-w-[44rem] border-collapse text-xs">
        <thead>
          <tr>
            <th className="eyebrow py-2 pr-3 text-left font-medium">
              chosen for &darr;
            </th>
            {data.measured_in.map((label) => (
              <th
                key={label}
                className="eyebrow px-1 py-2 text-center font-medium"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.chosen_for.map((row, i) => (
            <tr key={row}>
              <td className="py-1 pr-3 text-[0.8125rem] whitespace-nowrap text-ink">
                {row}
              </td>
              {data.sharpe[i].map((value, j) => {
                const diagonal = i === j;
                const style = shade(value);
                return (
                  <td key={j} className="p-0.5">
                    <div
                      className="tabular py-1.5 text-center"
                      style={{
                        ...style,
                        outline: diagonal
                          ? "1px solid var(--color-ink)"
                          : undefined,
                      }}
                      title={
                        diagonal
                          ? "In-sample: chosen knowing what happened here"
                          : `${row} allocation, measured in ${data.measured_in[j]}`
                      }
                    >
                      {num(value, 2)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Periods({ result }: { result: PeriodsResult }) {
  const stability = [...result.stability].sort((a, b) => b.spread - a.spread);
  const leastStable = stability[0];
  const mostStable = stability[stability.length - 1];

  return (
    <div className="space-y-5">
      <Panel>
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-3">
          <Stat label="regimes compared" value={String(result.by_period.length)} />
          <Stat
            label="average hindsight premium"
            value={`${num(result.average_premium, 2)} Sharpe`}
          />
          <Stat
            label="least settled asset"
            value={leastStable ? assetLabel(leastStable.asset) : "--"}
          />
        </div>
        <p className="mt-5 border-t border-line pt-4 text-sm leading-relaxed text-muted">
          The hindsight premium is how much better each period's own allocation
          did there than allocations chosen for other periods. It is the value
          of having known the answer in advance, and nobody does — so it is
          also the amount by which any single-period result flatters itself.
        </p>
      </Panel>

      <Panel title="each period's winner, measured everywhere else">
        <Matrix data={result.cross_period} />
        <p className="mt-4 text-xs leading-relaxed text-muted">
          Outlined cells are in-sample and win by construction. Read across a
          row instead: that is the same allocation facing conditions it was not
          chosen for. Petrol is a positive Sharpe, brick a negative one.
        </p>
      </Panel>

      <Panel title="how much each answer depends on the period">
        <div className="space-y-2.5">
          {stability.map((row) => (
            <div
              key={row.asset}
              className="grid grid-cols-[9.5rem_1fr_5rem] items-center gap-4"
            >
              <span className="truncate text-[0.8125rem] text-ink">
                {assetLabel(row.asset)}
              </span>
              <div className="relative h-5">
                <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-line" />
                <div
                  className="absolute top-1/2 h-2 -translate-y-1/2 bg-primary"
                  style={{
                    left: `${row.min * 100}%`,
                    width: `${Math.max(row.spread * 100, 0.4)}%`,
                    opacity: 0.28,
                  }}
                />
              </div>
              <span className="tabular text-right text-xs text-muted">
                {pct(row.spread, 0)}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-muted">
          An asset whose weight swings between regimes is not something the data
          has an opinion about — it is something the period has an opinion
          about. {leastStable && mostStable && (
            <>
              Here {assetLabel(leastStable.asset).toLowerCase()} moves most
              ({pct(leastStable.spread, 0)}) and{" "}
              {assetLabel(mostStable.asset).toLowerCase()} least (
              {pct(mostStable.spread, 0)}).
            </>
          )}
        </p>
      </Panel>

      <Panel title="best allocation, by period">
        <div className="-mx-5 overflow-x-auto px-5">
          <table className="w-full min-w-[46rem] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line-strong">
                <th className="eyebrow py-2 pr-4 text-left font-medium">period</th>
                {result.stability.map((row) => (
                  <th
                    key={row.asset}
                    className="eyebrow py-2 pr-4 text-right font-medium"
                  >
                    {assetLabel(row.asset).split(" ")[0]}
                  </th>
                ))}
                <th className="eyebrow py-2 pr-4 text-right font-medium">return</th>
                <th className="eyebrow py-2 text-right font-medium">drawdown</th>
              </tr>
            </thead>
            <tbody>
              {result.by_period.map((row) => (
                <tr key={row.period} className="border-b border-line last:border-b-0">
                  <td className="py-2 pr-4 whitespace-nowrap text-ink">
                    {row.period}
                  </td>
                  {result.stability.map((asset) => (
                    <td
                      key={asset.asset}
                      className="tabular py-2 pr-4 text-right text-muted"
                    >
                      {pct(row[asset.asset] as number, 0)}
                    </td>
                  ))}
                  <td className="tabular py-2 pr-4 text-right text-ink">
                    {pct(row.return, 1)}
                  </td>
                  <td className="tabular py-2 text-right text-brick">
                    {pct(row.max_dd, 1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="the allocation no period voted against">
        <div className="flex flex-wrap gap-x-8 gap-y-3">
          {Object.entries(result.consensus).map(([asset, weight]) => (
            <div key={asset}>
              <div className="eyebrow mb-1">{assetLabel(asset)}</div>
              <div className="tabular text-lg text-ink">{pct(weight)}</div>
            </div>
          ))}
        </div>
        <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-muted">
          The period answers averaged, weighted by length. Not an optimum for
          anything and not offered as one — a starting point for when the
          regimes disagree, which they usually do.
        </p>
      </Panel>
    </div>
  );
}
