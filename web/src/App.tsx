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
import {
  api,
  type Allocation,
  type CostBody,
  type MandateBody,
  type TrackBody,
} from "@/lib/api";
import {
  contains,
  identify,
  labelFor,
  toSaved,
  toggle,
  type Saved,
} from "@/lib/basket";
import { dateLabel } from "@/lib/format";
import { INITIAL, MandateForm, asFraction, type FormState } from "@/components/MandateForm";
import { Results } from "@/components/Results";
import { RegimeStress } from "@/components/RegimeStress";
import { ConstraintCost } from "@/components/ConstraintCost";
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

function buildRequest(
  state: FormState,
  rankBy: string,
  allAssets: string[],
  resolution: number | null,
): MandateBody {
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
    resolution,
  };
}

type View = "mandate" | "periods" | "cost";

export default function App() {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [rankBy, setRankBy] = useState("max_drawdown");
  // Allocations closer than this are the same portfolio to anyone deciding.
  // Five percentage points is the default because that is roughly the
  // granularity a committee argues at; one point is available for a closer
  // look, and null shows the raw qualifying set.
  const [resolution, setResolution] = useState<number | null>(0.05);
  const [view, setView] = useState<View>("mandate");
  // Allocations kept aside to compare. Held by their weights, not by row
  // position: reordering the table or solving a different mandate would
  // otherwise leave the ticks pointing at whatever now occupies that row.
  const [basket, setBasket] = useState<Saved[]>([]);
  const [regimes, setRegimes] = useState<string[]>([]);

  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });

  const solve = useMutation({
    mutationFn: (body: MandateBody) => api.mandate(body),
  });

  const cost = useMutation({
    mutationFn: (body: CostBody) => api.constraintCost(body),
  });

  const track = useMutation({
    mutationFn: (body: TrackBody) => api.track(body),
  });

  const allocatable = meta.data?.assets.filter((a) => a.allocatable).map((a) => a.key) ?? [];
  const inPlay = form.assets.length === 0 ? allocatable : form.assets;

  const run = (nextRank = rankBy) => {
    if (!meta.data) return;
    solve.mutate(buildRequest(form, nextRank, allocatable, resolution));
  };

  const changeRank = (value: string) => {
    setRankBy(value);
    if (solve.data?.feasible) run(value);
  };

  const changeResolution = (value: number | null) => {
    setResolution(value);
    if (!meta.data || !solve.data?.feasible) return;
    solve.mutate(buildRequest(form, rankBy, allocatable, value));
  };

  const isSaved = (allocation: Allocation) =>
    contains(basket, identify(
      Object.fromEntries(inPlay.map((a) => [a, (allocation[a] as number) ?? 0])),
      inPlay,
    ));

  const toggleSaved = (allocation: Allocation, index: number) =>
    setBasket((current) =>
      toggle(current, toSaved(allocation, inPlay, labelFor(rankBy, index))),
    );

  /** Measure the kept allocations through each regime.
   *
   * These are candidates that met a mandate, not per-period optima. A
   * period-by-period optimum is a corner solution nobody would hold, so what
   * it endured is not informative; what a real candidate endured is.
   */
  const runTracking = (
    saved: Saved[] = basket,
    periods: string[] = regimes,
  ) => {
    if (!meta.data) return;

    // Nothing kept yet: fall back to the top few from the current solve, so
    // the view is never blank on arrival.
    const allocations =
      saved.length > 0
        ? saved.map((entry) => ({ label: entry.label, weights: entry.weights }))
        : (solve.data?.allocations ?? []).slice(0, 5).map((row, i) => ({
            label: labelFor(rankBy, i),
            weights: Object.fromEntries(
              inPlay.map((asset) => [asset, (row[asset] as number) ?? 0]),
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

  const changeRegimes = (next: string[]) => {
    setRegimes(next);
    runTracking(basket, next);
  };

  /** What the policy limits cost, using the same limits the mandate uses.
   *
   * Slower than a single solve: it optimises with the rules and without them,
   * then once more per rule to isolate each. The sample budget is lower for
   * that reason -- the figure it produces is a difference between two solves,
   * and a difference is less sensitive to search depth than either solve is.
   */
  const runCost = () => {
    if (!meta.data) return;
    const request = buildRequest(form, rankBy, allocatable, resolution);

    if (
      Object.keys(request.constraints.caps ?? {}).length === 0 &&
      Object.keys(request.constraints.floors ?? {}).length === 0 &&
      Object.keys(request.constraints.group_caps ?? {}).length === 0
    ) {
      return;
    }

    cost.mutate({
      gold_weight: request.gold_weight,
      assets: request.assets,
      rebalance: request.rebalance,
      cost_bps: request.cost_bps,
      samples: form.rebalance === "monthly" ? 3000 : 600,
      objective: "max_sharpe",
      constraints: request.constraints,
      per_rule: true,
    });
  };

  const hasLimits =
    Object.values(form.limits).some(
      (limit) => limit.floor.trim() !== "" || limit.cap.trim() !== "",
    ) || form.growthCap.trim() !== "";

  const switchTo = (next: View) => {
    setView(next);
    if (next === "periods" && (basket.length > 0 || solve.data?.feasible)) {
      runTracking();
    }
    if (next === "cost" && cost.isIdle && hasLimits) runCost();
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
                  ["cost", "Cost of the rules"],
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
                  resolution={resolution}
                  onResolutionChange={changeResolution}
                  isSaved={isSaved}
                  onToggle={toggleSaved}
                  savedCount={basket.length}
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
                disabled={
                  track.isPending || (basket.length === 0 && !solve.data?.feasible)
                }
              >
                {track.isPending ? "Measuring…" : "Run again"}
              </Button>
            </div>

            {basket.length === 0 && !solve.data?.feasible && (
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
                samples={(basket.length || 5) * 7}
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

            {basket.length > 0 && (
              <div className="mb-5 flex flex-wrap items-center gap-2 border border-line bg-surface px-4 py-3">
                <span className="eyebrow mr-1">kept</span>
                {basket.map((entry) => (
                  <button
                    key={entry.id}
                    type="button"
                    onClick={() => {
                      const next = basket.filter((s) => s.id !== entry.id);
                      setBasket(next);
                      runTracking(next, regimes);
                    }}
                    title="Remove from the comparison"
                    className="border border-primary bg-primary-soft px-2 py-0.5 text-xs text-primary hover:border-brick hover:bg-brick-soft hover:text-brick"
                  >
                    {entry.label} &times;
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => setBasket([])}
                  className="ml-auto text-xs text-muted underline underline-offset-2 hover:text-ink"
                >
                  clear
                </button>
              </div>
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

        {meta.data && view === "cost" && (
          <div>
            <div className="mb-5 flex flex-wrap items-baseline justify-between gap-4">
              <p className="max-w-2xl text-[0.9375rem] leading-relaxed text-muted">
                The policy limits from the mandate, solved with and without
                them. Every rule costs return; almost nobody measures how much,
                and usually only one of them is doing it.
              </p>
              <Button
                type="button"
                variant="quiet"
                onClick={runCost}
                disabled={cost.isPending || !hasLimits}
              >
                {cost.isPending ? "Costing…" : "Run again"}
              </Button>
            </div>

            {!hasLimits && (
              <Panel title="no limits to cost">
                <p className="max-w-lg text-[0.9375rem] leading-relaxed text-ink">
                  Set a floor or a ceiling on the mandate tab first — a minimum
                  cash holding, a cap on private equity, a limit on growth
                  assets.
                </p>
                <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted">
                  This view then solves the allocation twice, once obeying those
                  rules and once ignoring them, and reports the difference.
                </p>
              </Panel>
            )}

            {cost.isPending && (
              <Solving
                samples={(form.rebalance === "monthly" ? 3000 : 600) * 4}
                schedule={form.rebalance}
              />
            )}

            {cost.isError && (
              <Panel title="the request was rejected">
                <p className="text-sm leading-relaxed text-brick">
                  {(cost.error as Error).message}
                </p>
              </Panel>
            )}

            {cost.data && !cost.isPending && <ConstraintCost result={cost.data} />}
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
