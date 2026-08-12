/** What the mandate produced.
 *
 * Two outcomes, and the interface treats them as equally legitimate. A
 * mandate that cannot be met is not a failed request -- it is a finding, and
 * often the more useful one. So the infeasible state is designed rather than
 * being an error message: it names each thing that would have to change and by
 * how much, and the user chooses which rule to argue with.
 */

import type { Allocation, MandateResult } from "@/lib/api";
import { assetLabel, months, num, pct } from "@/lib/format";
import { Panel, Select, Stat } from "@/components/ui/primitives";
import { Envelope } from "@/components/Envelope";

const MEASURES: Record<string, string> = {
  max_drawdown: "shallowest drawdown",
  realised_return: "highest return",
  volatility: "lowest volatility",
  sharpe: "best Sharpe",
  sortino: "best Sortino",
  months_to_recover: "fastest recovery",
  months_underwater: "least time underwater",
};

export function Infeasible({ result }: { result: MandateResult }) {
  return (
    <Panel title="not achievable">
      <p className="max-w-2xl text-[0.9375rem] leading-relaxed text-ink">
        Nothing in {result.n_sampled.toLocaleString()} sampled allocations met
        every requirement over this period. Any one of these changes would make
        it reachable.
      </p>

      <ul className="mt-5 space-y-0">
        {(result.relaxations ?? []).map((relaxation, index) => (
          <li
            key={index}
            className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-t border-line py-3.5 first:border-t-0 first:pt-0"
          >
            <span className="w-44 shrink-0 text-[0.8125rem] text-ink">
              {relaxation.what}
            </span>
            <span className="tabular text-sm text-brick">
              {pct(relaxation.current)} &rarr; {pct(relaxation.required)}
            </span>
            {relaxation.note && (
              <span className="text-xs text-muted">{relaxation.note}</span>
            )}
          </li>
        ))}
      </ul>

      {(result.relaxations ?? []).length === 0 && (
        <p className="mt-4 text-sm text-muted">
          No single change is enough — the requirements conflict with each
          other rather than with the data.
        </p>
      )}
    </Panel>
  );
}

function AllocationTable({
  allocations,
  assets,
}: {
  allocations: Allocation[];
  assets: string[];
}) {
  return (
    <div className="-mx-5 overflow-x-auto px-5">
      <table className="w-full min-w-[52rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line-strong">
            {assets.map((asset) => (
              <th
                key={asset}
                className="eyebrow py-2 pr-4 text-right font-medium first:text-left"
              >
                {assetLabel(asset).split(" ")[0]}
              </th>
            ))}
            <th className="eyebrow py-2 pr-4 text-right font-medium">return</th>
            <th className="eyebrow py-2 pr-4 text-right font-medium">vol</th>
            <th className="eyebrow py-2 pr-4 text-right font-medium">drawdown</th>
            <th className="eyebrow py-2 pr-4 text-right font-medium">sharpe</th>
            <th className="eyebrow py-2 text-right font-medium">recovery</th>
          </tr>
        </thead>
        <tbody>
          {allocations.map((allocation, index) => (
            <tr
              key={index}
              className="border-b border-line last:border-b-0 hover:bg-ground"
            >
              {assets.map((asset) => (
                <td key={asset} className="tabular py-2 pr-4 text-right text-muted">
                  {pct(allocation[asset] as number, 0)}
                </td>
              ))}
              <td className="tabular py-2 pr-4 text-right text-ink">
                {pct(allocation.realised_return, 2)}
              </td>
              <td className="tabular py-2 pr-4 text-right text-muted">
                {pct(allocation.volatility, 2)}
              </td>
              <td className="tabular py-2 pr-4 text-right text-brick">
                {pct(allocation.max_drawdown, 1)}
              </td>
              <td className="tabular py-2 pr-4 text-right text-muted">
                {num(allocation.sharpe)}
              </td>
              <td className="tabular py-2 text-right text-muted">
                {months(allocation.months_to_recover)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Results({
  result,
  assets,
  rankBy,
  onRankChange,
  rankable,
}: {
  result: MandateResult;
  assets: string[];
  rankBy: string;
  onRankChange: (value: string) => void;
  rankable: string[];
}) {
  if (!result.feasible) return <Infeasible result={result} />;

  const best = result.allocations?.[0];
  const share = result.n_qualifying / result.n_sampled;

  return (
    <div className="space-y-5">
      <Panel>
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
          <Stat
            label="allocations that qualify"
            value={result.n_qualifying.toLocaleString()}
          />
          <Stat label="of those sampled" value={pct(share, 1)} />
          <Stat label="best return" value={pct(best?.realised_return, 2)} />
          <Stat
            label="its worst drawdown"
            value={pct(best?.max_drawdown, 1)}
            tone="brick"
          />
        </div>
      </Panel>

      <Panel title="the range of answers">
        <Envelope rows={result.envelope ?? []} best={best} />
      </Panel>

      {result.sleeve_split && (
        <Panel title="what the commodities sleeve holds">
          <div className="flex flex-wrap items-baseline gap-x-8 gap-y-3">
            <div>
              <div className="eyebrow mb-1">sleeve</div>
              <div className="tabular text-lg text-ink">
                {pct(result.sleeve_split.sleeve_weight)}
              </div>
            </div>
            <span className="text-muted">splits into</span>
            <div>
              <div className="eyebrow mb-1">gold</div>
              <div className="tabular text-lg" style={{ color: "var(--color-brass)" }}>
                {pct(result.sleeve_split.gold)}
              </div>
            </div>
            <div>
              <div className="eyebrow mb-1">commodities ex-gold</div>
              <div className="tabular text-lg text-ink">
                {pct(result.sleeve_split.commodities_ex_gold)}
              </div>
            </div>
          </div>

          <div className="mt-4 flex h-2 overflow-hidden border border-line">
            <div
              style={{
                width: `${result.sleeve_split.gold_weight * 100}%`,
                background: "var(--color-brass)",
              }}
            />
            <div
              style={{
                width: `${(1 - result.sleeve_split.gold_weight) * 100}%`,
                background: "var(--color-primary)",
                opacity: 0.35,
              }}
            />
          </div>

          <p className="mt-3 text-xs leading-relaxed text-muted">
            For the allocation ranked first, at a{" "}
            {pct(result.sleeve_split.gold_weight, 0)} gold setting. The split is
            the same proportion for every qualifying allocation — only the total
            changes.
          </p>
        </Panel>
      )}

      <Panel
        title="qualifying allocations"
        aside={
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">ranked by</span>
            <Select
              value={rankBy}
              onChange={(e) => onRankChange(e.target.value)}
              className="w-auto py-1 text-xs"
            >
              {rankable.map((measure) => (
                <option key={measure} value={measure}>
                  {MEASURES[measure] ?? measure}
                </option>
              ))}
            </Select>
          </div>
        }
      >
        <AllocationTable allocations={result.allocations ?? []} assets={assets} />
        <p className="mt-4 text-xs text-muted">
          Showing {result.allocations?.length ?? 0} of{" "}
          {result.n_qualifying.toLocaleString()}. They all meet the mandate —
          which is best depends on what you care about, so change the ranking
          rather than trusting the first row.
        </p>
      </Panel>
    </div>
  );
}
