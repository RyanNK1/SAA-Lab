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
import { api, type MandateBody, type TrackBody } from "@/lib/api";
import { dateLabel } from "@/lib/format";
import { INITIAL, MandateForm, asFraction, type FormState } from "@/components/MandateForm";
import { Results } from "@/components/Results";
import { RegimeStress } from "@/components/RegimeStress";
import { Solving } from "@/components/Solving";
import { Button, Panel } from "@/components/ui/primitives";

const LIMIT = 12;

/** How many allocations to measure.
 *
 * Monthly rebalancing is measured in one vectorised pass, so the budget can be
 * generous. Every other schedule simulates each allocation separately -- about
 * two orders of magnitude slower per allocation -- and 8,000 of those would
 * take minutes, which is not a wait anyone should sit through to move a
 * slider. The budget drops accordingly, and the interface says so rather than
 * quietly returning a coarser answer.
 */
export const sampleBudget = (schedule: string): number =>
  schedule === "monthly" ? 8000 : 1500;

function buildRequest(state: FormState, rankBy: string, allAssets: string[]): MandateBody {
  const caps: Record<string, number> = {};
  const floors: Record<string, number> = {};
  const groupCaps: Record<string, number> = {};

  const chosen = state.assets.length === 0 ? allAssets : state.assets;

  // A limit on an asset the user has deselected would be rejected as unknown,
  // so only the buckets actually in play are sent.
  for (const key of chosen) {
    const limit = state.limits[key];
    if (!limit) continue;
    const floor = asFraction(limit.floor);
    const cap = asFraction(limit.cap);
    if (floor !== null) floors[key] = floor;
    if (cap !== null) caps[key] = cap;
  }

  const growthCap = asFraction(state.growthCap);
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
    samples: sampleBudget(state.rebalance),
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

type View = "mandate" | "periods";

export default function App() {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [rankBy, setRankBy] = useState("max_drawdown");
  const [view, setView] = useState<View>("mandate");
  // Which qualifying allocations to carry into the regime view. Empty means
  // the top few, so the second view is never blank on arrival.
  const [tracked, setTracked] = useState<number[]>([]);
  const [regimes, setRegimes] = useState<string[]>([]);

  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });

  const solve = useMutation({
    mutationFn: (body: MandateBody) => api.mandate(body),
  });

  const track = useMutation({
    mutationFn: (body: TrackBody) => api.track(body),
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

  /** Carry the chosen allocations into the regime view.
   *
   * These are candidates that met a mandate, not per-period optima. A
   * period-by-period optimum is a corner solution nobody would hold, so what
   * it endured is not informative; what a real candidate endured is.
   */
  const runTracking = (
    indices: number[] = tracked,
    periods: string[] = regimes,
  ) => {
    if (!meta.data || !solve.data?.allocations) return;

    const rows = solve.data.allocations;
    // Default to the first few rather than requiring a selection first: an
    // empty second view would make the user guess what it is for.
    const chosen = indices.length > 0 ? indices : rows.slice(0, 5).map((_, i) => i);

    const allocations = chosen
      .filter((i) => rows[i])
      .map((i) => ({
        label: `#${i + 1}`,
        weights: Object.fromEntries(
          inPlay.map((asset) => [asset, (rows[i][asset] as number) ?? 0]),
        ),
      }));

    if (allocations.length === 0) return;

    track.mutate({
      gold_weight: form.goldWeight,
      assets: form.assets.length === 0 ? undefined : form.assets,
      rebalance: form.rebalance,
      cost_bps: Number(form.costBps) || 0,
      samples: 100,
      allocations,
      periods: periods.length > 0 ? periods : undefined,
    });
  };

  const toggleTracked = (index: number) =>
    setTracked((current) =>
      current.includes(index)
        ? current.filter((i) => i !== index)
        : [...current, index],
    );

  const changeRegimes = (next: string[]) => {
    setRegimes(next);
    runTracking(tracked, next);
  };

  const switchTo = (next: View) => {
    setView(next);
    if (next === "periods" && solve.data?.feasible) runTracking();
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
            Result in hindsight, not a forecast.
          </p>

          {meta.data && (
            <nav className="mt-6 flex gap-6 border-b border-line" aria-label="Views">
              {(
                [
                  ["mandate", "One mandate"],
                  ["periods", "Across regimes"],
                ] as [View, string][]
              ).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => switchTo(key)}
                  aria-current={view === key ? "page" : undefined}
                  className={
                    view === key
                      ? "-mb-px border-b-2 border-primary pb-2.5 text-sm font-medium text-ink"
                      : "-mb-px border-b-2 border-transparent pb-2.5 text-sm text-muted hover:text-ink"
                  }
                >
                  {label}
                </button>
              ))}
            </nav>
          )}
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

        {meta.data && view === "mandate" && (
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
                <Solving
                  samples={sampleBudget(form.rebalance)}
                  schedule={form.rebalance}
                />
              )}

              {solve.isError && (
                <Panel title="the request was rejected">
                  <p className="text-sm leading-relaxed text-brick">
                    {(solve.error as Error).message}
                  </p>
                </Panel>
              )}

              {solve.data && !solve.isPending && (
                <Results
                  result={solve.data}
                  assets={inPlay}
                  rankBy={rankBy}
                  onRankChange={changeRank}
                  rankable={meta.data.rankable}
                  selected={tracked}
                  onToggle={toggleTracked}
                />
              )}
            </div>
          </div>
        )}

        {meta.data && view === "periods" && (
          <div>
            <div className="mb-5 flex flex-wrap items-baseline justify-between gap-4">
              <p className="max-w-2xl text-[0.9375rem] leading-relaxed text-muted">
                The allocations you ticked, measured through each regime
                separately. These met the mandate over the whole sample — this
                is what they endured along the way.
              </p>
              <Button
                type="button"
                variant="quiet"
                onClick={() => runTracking()}
                disabled={track.isPending || !solve.data?.feasible}
              >
                {track.isPending ? "Measuring…" : "Run again"}
              </Button>
            </div>

            {!solve.data?.feasible && (
              <Panel title="nothing to track yet">
                <p className="max-w-lg text-[0.9375rem] leading-relaxed text-ink">
                  Solve a mandate first. This view takes the allocations that
                  met it and shows how each one behaved through the crisis, the
                  recovery, the inflation shock and the rest.
                </p>
                <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted">
                  It deliberately does not optimise per regime. The best
                  allocation for a crisis, chosen knowing the crisis happened,
                  is a corner solution nobody would hold — so what it endured
                  tells you nothing.
                </p>
              </Panel>
            )}

            {track.isPending && (
              <Solving
                samples={(tracked.length || 5) * 7}
                schedule={form.rebalance}
              />
            )}

            {track.isError && (
              <Panel title="the request was rejected">
                <p className="text-sm leading-relaxed text-brick">
                  {(track.error as Error).message}
                </p>
              </Panel>
            )}

            {track.data && !track.isPending && (
              <RegimeStress
                result={track.data}
                selectedPeriods={
                  regimes.length > 0
                    ? regimes
                    : track.data.periods.map((p) => p.label)
                }
                onPeriodsChange={changeRegimes}
                allPeriods={meta.data.regimes}
              />
            )}
          </div>
        )}

      </main>

      <footer className="mt-12 border-t border-line bg-surface">
        <div className="mx-auto max-w-6xl px-6 py-7">
          <h2 className="eyebrow mb-3">what each asset class is</h2>
          <dl className="grid gap-x-8 gap-y-2 text-xs leading-relaxed sm:grid-cols-2">
            {meta.data?.assets.map((asset) => (
              <div key={asset.key} className="flex gap-3">
                <dt className="w-28 shrink-0 text-ink">{asset.label}</dt>
                <dd className="text-muted">{asset.proxy}</dd>
              </div>
            ))}
            {meta.data?.sleeve.components.map((component) => (
              <div key={component.key} className="flex gap-3">
                <dt className="w-28 shrink-0 pl-3 text-muted">
                  &mdash; {component.label}
                </dt>
                <dd className="text-muted">{component.proxy}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-4 border-t border-line pt-4 text-xs leading-relaxed text-muted">
            Proxies, not the institutional instruments themselves. Private
            equity is stood in for by small-cap equity, which correlates around
            0.9 with public equity — so that sleeve is not the diversifier its
            label suggests.
          </p>
        </div>
      </footer>
    </div>
  );
}
