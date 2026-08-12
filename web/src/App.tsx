/** The application shell.
 *
 * One screen, one question: state a mandate, see which allocations met it.
 *
 * The header carries the disclaimer permanently rather than hiding it in a
 * footnote. Every figure here is hindsight -- what would have been best over a
 * chosen window, knowing what happened in it -- and an interface that presents
 * that as a recommendation would be lying by omission. It is the first thing
 * on the page for that reason.
 */

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, type MandateBody } from "@/lib/api";
import { dateLabel } from "@/lib/format";
import { INITIAL, MandateForm, asFraction, type FormState } from "@/components/MandateForm";
import { Results } from "@/components/Results";
import { Panel } from "@/components/ui/primitives";

const SAMPLES = 8000;
const LIMIT = 12;

function buildRequest(state: FormState, rankBy: string, allAssets: string[]): MandateBody {
  const caps: Record<string, number> = {};
  const floors: Record<string, number> = {};
  const groupCaps: Record<string, number> = {};

  const peCap = asFraction(state.peCap);
  const cashFloor = asFraction(state.cashFloor);
  const growthCap = asFraction(state.growthCap);

  const chosen = state.assets.length === 0 ? allAssets : state.assets;

  // A limit on an asset the user has deselected would be rejected as unknown,
  // so only send the ones that are actually in play.
  if (peCap !== null && chosen.includes("private_equity")) caps.private_equity = peCap;
  if (cashFloor !== null && chosen.includes("cash")) floors.cash = cashFloor;
  if (growthCap !== null && chosen.some((a) => a === "equity" || a === "private_equity")) {
    groupCaps.growth = growthCap;
  }

  return {
    start: state.start || undefined,
    end: state.end || undefined,
    gold_weight: state.goldWeight,
    assets: state.assets.length === 0 ? undefined : state.assets,
    rebalance: state.rebalance,
    cost_bps: Number(state.costBps) || 0,
    samples: SAMPLES,
    target_return: asFraction(state.targetReturn),
    max_volatility: asFraction(state.maxVolatility),
    max_drawdown: (() => {
      const value = asFraction(state.maxDrawdown);
      // Entered as a positive figure because that is how people say it, and
      // sent negative because that is what a drawdown is.
      return value === null ? null : -Math.abs(value);
    })(),
    max_recovery_months: state.maxRecovery.trim() === "" ? null : Number(state.maxRecovery),
    constraints: { caps, floors, group_caps: groupCaps },
    rank_by: rankBy,
    limit: LIMIT,
  };
}

export default function App() {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [rankBy, setRankBy] = useState("max_drawdown");

  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });

  const solve = useMutation({
    mutationFn: (body: MandateBody) => api.mandate(body),
  });

  const allocatable = meta.data?.assets.filter((a) => a.allocatable).map((a) => a.key) ?? [];
  const inPlay = form.assets.length === 0 ? allocatable : form.assets;

  const run = (nextRank = rankBy) => {
    if (!meta.data) return;
    solve.mutate(buildRequest(form, nextRank, allocatable));
  };

  const changeRank = (value: string) => {
    setRankBy(value);
    if (solve.data?.feasible) run(value);
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto max-w-6xl px-6 py-8">
          <div className="flex flex-wrap items-end justify-between gap-6">
            <div>
              <h1
                className="text-3xl font-semibold tracking-tight text-ink"
                style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}
              >
                SAA Lab
              </h1>
              <p className="mt-2 max-w-xl text-[0.9375rem] leading-relaxed text-muted">
                State what an allocation has to achieve and what it may not
                exceed. The answer is every split of the money that would have
                met it.
              </p>
            </div>

            {meta.data && (
              <dl className="tabular text-right text-xs text-muted">
                <dt className="eyebrow mb-1">covering</dt>
                <dd>
                  {dateLabel(meta.data.coverage.start)} &ndash;{" "}
                  {dateLabel(meta.data.coverage.end)}
                </dd>
                <dd>{meta.data.coverage.months} months, USD</dd>
              </dl>
            )}
          </div>

          <p className="mt-6 border-l-2 border-brass pl-3 text-xs leading-relaxed text-muted">
            Every result is hindsight: what would have been best over the period
            chosen, knowing what happened in it. That is exactly answerable, and
            it is not a forecast.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {meta.isLoading && (
          <p className="text-sm text-muted">Loading the dataset…</p>
        )}

        {meta.isError && (
          <Panel title="the api is not answering">
            <p className="text-sm leading-relaxed text-ink">
              {(meta.error as Error).message}
            </p>
            <p className="mt-3 text-sm text-muted">
              Start it with{" "}
              <code className="tabular bg-ground px-1.5 py-0.5">
                python scripts/serve.py
              </code>{" "}
              and reload.
            </p>
          </Panel>
        )}

        {meta.data && (
          <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
            <div className="lg:sticky lg:top-6 lg:self-start">
              <MandateForm
                meta={meta.data}
                state={form}
                onChange={setForm}
                onSubmit={() => run()}
                pending={solve.isPending}
              />
            </div>

            <div>
              {solve.isIdle && (
                <Panel title="nothing solved yet">
                  <p className="max-w-lg text-[0.9375rem] leading-relaxed text-ink">
                    The mandate on the left asks for 6% a year at no more than
                    10% volatility, holding at least 5% cash and no more than
                    20% private equity.
                  </p>
                  <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted">
                    Change what you need it to do, then find the allocations
                    that would have done it.
                  </p>
                </Panel>
              )}

              {solve.isPending && (
                <Panel title="solving">
                  <p className="text-sm text-muted">
                    Measuring {SAMPLES.toLocaleString()} allocations over the
                    period.
                  </p>
                </Panel>
              )}

              {solve.isError && (
                <Panel title="the request was rejected">
                  <p className="text-sm leading-relaxed text-brick">
                    {(solve.error as Error).message}
                  </p>
                </Panel>
              )}

              {solve.data && (
                <Results
                  result={solve.data}
                  assets={inPlay}
                  rankBy={rankBy}
                  onRankChange={changeRank}
                  rankable={meta.data.rankable}
                />
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="mt-12 border-t border-line">
        <div className="mx-auto max-w-6xl px-6 py-6 text-xs leading-relaxed text-muted">
          Built from public price data. Private equity is proxied by small-cap
          equity, which correlates around 0.9 with public equity — so that
          sleeve is not the diversifier its label suggests.
        </div>
      </footer>
    </div>
  );
}
