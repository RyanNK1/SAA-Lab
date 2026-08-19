/** What the policy limits cost.
 *
 * Every rule an investment policy statement imposes costs return, and almost
 * nobody measures how much. Reporting it turns the tool from "here is the
 * answer" into "here is what your rules are costing you", which is the
 * question a committee actually has -- and the one that makes a debate about
 * a cap resolvable rather than a matter of taste.
 *
 * The per-rule figures are the useful part. A policy statement usually has
 * several limits and typically only one of them binds; knowing which turns an
 * argument about all of them into an argument about one.
 *
 * A rule that costs nothing is a real answer, not a failed calculation.
 * "Your 20% alternatives cap has cost you nothing in twenty years" is worth
 * hearing, and the panel says it plainly rather than showing a blank.
 */

import type { CostResult } from "@/lib/api";
import { assetLabel, pct } from "@/lib/format";
import { Panel } from "@/components/ui/primitives";

const KIND_LABEL: Record<string, string> = {
  cap: "ceiling",
  floor: "floor",
  group: "group limit",
};

export function ConstraintCost({ result }: { result: CostResult }) {
  const rules = result.per_rule ?? [];
  const binding = rules.filter((rule) => Math.abs(rule.cost_bps) > 0.5);
  const free = rules.filter((rule) => Math.abs(rule.cost_bps) <= 0.5);
  const widest = Math.max(...rules.map((r) => Math.abs(r.cost_bps)), 1);

  return (
    <div className="space-y-5">
      <Panel title="what the rules cost">
        <div className="flex flex-wrap items-baseline gap-x-10 gap-y-4">
          <div>
            <div className="eyebrow mb-1">total, per year</div>
            <div
              className="tabular text-2xl"
              style={{
                color: result.binding ? "var(--color-brick)" : "var(--color-ink)",
              }}
            >
              {result.return_cost_bps < 0.5
                ? "nothing"
                : `${result.return_cost_bps.toFixed(0)} bps`}
            </div>
          </div>
          <div>
            <div className="eyebrow mb-1">unconstrained return</div>
            <div className="tabular text-lg text-ink">
              {pct(result.unconstrained.stats.realised_return, 2)}
            </div>
          </div>
          <div>
            <div className="eyebrow mb-1">with your limits</div>
            <div className="tabular text-lg text-ink">
              {pct(result.constrained.stats.realised_return, 2)}
            </div>
          </div>
        </div>

        <p className="mt-5 border-t border-line pt-4 text-sm leading-relaxed text-muted">
          {result.binding ? (
            <>
              Solving without the limits and with them, and taking the
              difference. The cost cannot be negative: a rule shrinks the set of
              allowed allocations, and a smaller set cannot contain a better
              answer.
            </>
          ) : (
            <>
              Nothing. The best allocation already satisfied every rule over
              this period, so none of them restricted anything — which is worth
              knowing before anyone argues about changing them.
            </>
          )}
        </p>
      </Panel>

      {rules.length > 0 && (
        <Panel title="which rule is doing it">
          <div className="space-y-3">
            {[...rules]
              .sort((a, b) => Math.abs(b.cost_bps) - Math.abs(a.cost_bps))
              .map((rule) => {
                const cost = Math.abs(rule.cost_bps);
                const isFree = cost <= 0.5;
                return (
                  <div
                    key={`${rule.kind}-${rule.constraint}`}
                    className="grid grid-cols-[11rem_1fr_5rem] items-center gap-4"
                  >
                    <span className="truncate text-[0.8125rem] text-ink">
                      {rule.constraint}
                      <span className="ml-1.5 text-[0.6875rem] text-muted">
                        {KIND_LABEL[rule.kind] ?? rule.kind}
                      </span>
                    </span>

                    <div className="relative h-5">
                      <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-line" />
                      {!isFree && (
                        <div
                          className="absolute top-1/2 h-2.5 -translate-y-1/2"
                          style={{
                            width: `${Math.max((cost / widest) * 100, 1)}%`,
                            background: "var(--color-brick)",
                            opacity: 0.4,
                          }}
                        />
                      )}
                    </div>

                    <span
                      className="tabular text-right text-xs"
                      style={{
                        color: isFree
                          ? "var(--color-muted)"
                          : "var(--color-brick)",
                      }}
                    >
                      {isFree ? "free" : `${cost.toFixed(0)} bps`}
                    </span>
                  </div>
                );
              })}
          </div>

          <p className="mt-5 border-t border-line pt-3 text-xs leading-relaxed text-muted">
            Each row removes one rule and solves again, so the figure is that
            rule&apos;s cost given the others. They do not sum to the total —
            rules interact, and two can be individually cheap and jointly
            expensive.
            {binding.length === 1 && free.length > 0 && (
              <>
                {" "}
                Here only <span className="text-ink">{binding[0].constraint}</span>{" "}
                binds; the other {free.length === 1 ? "rule costs" : `${free.length} rules cost`}{" "}
                nothing.
              </>
            )}
          </p>
        </Panel>
      )}

      <Panel title="what the limits changed">
        <div className="-mx-5 overflow-x-auto px-5">
          <table className="w-full min-w-[34rem] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line-strong">
                <th className="eyebrow py-2 pr-4 text-left font-medium">asset</th>
                <th className="eyebrow py-2 pr-4 text-right font-medium">
                  unconstrained
                </th>
                <th className="eyebrow py-2 pr-4 text-right font-medium">
                  constrained
                </th>
                <th className="eyebrow py-2 text-right font-medium">change</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(result.constrained.weights).map((asset) => {
                const free = result.unconstrained.weights[asset] ?? 0;
                const bound = result.constrained.weights[asset] ?? 0;
                const change = bound - free;
                return (
                  <tr key={asset} className="border-b border-line last:border-b-0">
                    <td className="py-2 pr-4 text-ink">{assetLabel(asset)}</td>
                    <td className="tabular py-2 pr-4 text-right text-muted">
                      {pct(free)}
                    </td>
                    <td className="tabular py-2 pr-4 text-right text-ink">
                      {pct(bound)}
                    </td>
                    <td
                      className="tabular py-2 text-right"
                      style={{
                        color:
                          Math.abs(change) < 0.005
                            ? "var(--color-muted)"
                            : "var(--color-ink)",
                      }}
                    >
                      {Math.abs(change) < 0.005
                        ? "—"
                        : `${change > 0 ? "+" : ""}${pct(change)}`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
