/** The frontier: the most return available at each level of risk.
 *
 * Every point is the best allocation for one level of ambition, so the shape
 * of the curve is the trade-off itself -- and the shape is the argument. It
 * bends: the first few percent of return are nearly free, and the last few are
 * ruinously expensive. Reading that off a table takes effort; seeing it takes
 * a second.
 *
 * The qualifying allocations from the mandate are drawn on top, which is what
 * makes this more than a textbook illustration. They sit below the curve --
 * they must, since the curve is the best attainable -- and how far below is
 * the honest measure of what the mandate's requirements and its policy limits
 * cost. A cloud hugging the line means the mandate is close to optimal; a
 * cloud well beneath it means the rules are expensive.
 *
 * Drawdown gets its own chart rather than a second axis on the first. Two
 * y-axes on one plot invite the reader to compare gradients that are not
 * comparable, and the point here is that the two measures diverge: over this
 * dataset the last stretch of return costs about five points of volatility
 * and twenty points of drawdown. Separate panels make that a comparison
 * rather than an optical illusion.
 */

import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Allocation, FrontierPoint } from "@/lib/api";
import { assetLabel, pct } from "@/lib/format";
import { Panel } from "@/components/ui/primitives";

const ASSETS = [
  "equity",
  "fixed_income",
  "private_equity",
  "commodities",
  "cash",
];

type Point = Record<string, number>;

function CurveTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: Point }[];
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  if (point.expected_return === undefined) return null;

  const held = ASSETS.filter((asset) => (point[asset] ?? 0) > 0.005).sort(
    (a, b) => (point[b] ?? 0) - (point[a] ?? 0),
  );

  return (
    <div className="border border-line-strong bg-surface px-3 py-2 text-xs shadow-[0_2px_12px_rgba(19,26,24,0.12)]">
      <div className="tabular mb-1 text-ink">
        {pct(point.expected_return, 2)} return at {pct(point.volatility, 2)}{" "}
        volatility
      </div>
      <div className="tabular mb-2" style={{ color: "var(--color-brick)" }}>
        worst fall {pct(point.max_drawdown, 1)}
      </div>
      {held.length > 0 && (
        <div className="tabular border-t border-line pt-1.5 text-[0.6875rem] text-muted">
          {held
            .map(
              (asset) =>
                `${assetLabel(asset).split(" ")[0]} ${pct(point[asset], 0)}`,
            )
            .join(" · ")}
        </div>
      )}
    </div>
  );
}

const axisLabel = (value: string) => ({
  value,
  position: "insideBottom" as const,
  offset: -14,
  style: {
    fontSize: 11,
    fill: "var(--color-muted)",
    letterSpacing: "0.1em",
    textTransform: "uppercase" as const,
  },
});

const tick = { fontSize: 11, fill: "var(--color-muted)" };
const asPercent = (v: number) => `${(v * 100).toFixed(0)}%`;

export function Frontier({
  points,
  qualifying,
  targetReturn,
  maxVolatility,
}: {
  points: FrontierPoint[];
  qualifying?: Allocation[];
  targetReturn?: number | null;
  maxVolatility?: number | null;
}) {
  if (points.length === 0) {
    return (
      <Panel title="the frontier">
        <p className="text-sm text-muted">
          No curve could be drawn for this selection. It usually means the
          weight limits leave only one feasible allocation, so there is no
          range of ambition to sweep.
        </p>
      </Panel>
    );
  }

  const cloud = (qualifying ?? []).map((allocation) => ({
    volatility: allocation.volatility,
    qualifying_return: allocation.realised_return,
  }));

  const deepest = Math.min(...points.map((p) => p.max_drawdown));

  // The cost of the last stretch of the curve, stated rather than left to be
  // inferred: it is the single most useful thing the chart shows.
  const midpoint = points[Math.floor(points.length / 2)];
  const furthest = points[points.length - 1];
  const returnGain = furthest.expected_return - midpoint.expected_return;
  const volCost = furthest.volatility - midpoint.volatility;
  const drawdownCost = midpoint.max_drawdown - furthest.max_drawdown;

  return (
    <div className="space-y-5">
      <Panel title="the frontier">
        <p className="mb-5 max-w-2xl text-sm leading-relaxed text-muted">
          The most return available at each level of risk. Nothing can sit above
          the line — it is the best that was attainable — so where the
          qualifying allocations fall below it is what the mandate and its
          limits cost.
        </p>

        <div className="h-[22rem] w-full">
          <ResponsiveContainer>
            <ComposedChart
              data={points as unknown as Point[]}
              margin={{ top: 8, right: 16, bottom: 28, left: 4 }}
            >
              <CartesianGrid stroke="var(--color-line)" strokeDasharray="2 4" />
              <XAxis
                type="number"
                dataKey="volatility"
                domain={[0, "dataMax"]}
                tickFormatter={asPercent}
                tick={tick}
                stroke="var(--color-line-strong)"
                label={axisLabel("volatility")}
              />
              <YAxis
                type="number"
                domain={[0, "dataMax"]}
                tickFormatter={asPercent}
                tick={tick}
                stroke="var(--color-line-strong)"
                width={44}
              />
              <Tooltip
                content={<CurveTooltip />}
                cursor={{ stroke: "var(--color-line-strong)", strokeWidth: 1 }}
              />

              {cloud.length > 0 && (
                <Scatter
                  data={cloud}
                  dataKey="qualifying_return"
                  fill="var(--color-brass)"
                  fillOpacity={0.5}
                  isAnimationActive={false}
                />
              )}

              <Line
                type="monotone"
                dataKey="expected_return"
                stroke="var(--color-primary)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 border-t border-line pt-3 text-xs text-muted">
          <span className="flex items-center gap-2">
            <span
              className="inline-block h-0.5 w-5"
              style={{ background: "var(--color-primary)" }}
            />
            best attainable
          </span>
          {cloud.length > 0 && (
            <span className="flex items-center gap-2">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: "var(--color-brass)", opacity: 0.5 }}
              />
              qualifying allocations ({cloud.length})
            </span>
          )}
          {(targetReturn || maxVolatility) && (
            <span className="tabular">
              mandate: {targetReturn ? `at least ${pct(targetReturn, 1)}` : "—"}{" "}
              at {maxVolatility ? `no more than ${pct(maxVolatility, 1)}` : "any"}{" "}
              volatility
            </span>
          )}
        </div>
      </Panel>

      <Panel title="what the last of the return costs">
        <p className="mb-4 max-w-2xl text-sm leading-relaxed text-muted">
          The same curve against drawdown rather than volatility. Reaching from{" "}
          <span className="tabular text-ink">
            {pct(midpoint.expected_return, 1)}
          </span>{" "}
          to{" "}
          <span className="tabular text-ink">
            {pct(furthest.expected_return, 1)}
          </span>{" "}
          — {pct(returnGain, 1)} more a year — costs {pct(volCost, 1)} of
          volatility and{" "}
          <span className="tabular" style={{ color: "var(--color-brick)" }}>
            {pct(drawdownCost, 1)}
          </span>{" "}
          of drawdown. The second is what anyone holding it actually
          experiences.
        </p>

        <div className="h-[16rem] w-full">
          <ResponsiveContainer>
            <ComposedChart
              data={points as unknown as Point[]}
              margin={{ top: 8, right: 16, bottom: 28, left: 4 }}
            >
              <CartesianGrid stroke="var(--color-line)" strokeDasharray="2 4" />
              <XAxis
                type="number"
                dataKey="expected_return"
                domain={["dataMin", "dataMax"]}
                tickFormatter={asPercent}
                tick={tick}
                stroke="var(--color-line-strong)"
                label={axisLabel("return")}
              />
              <YAxis
                type="number"
                domain={[Math.min(deepest * 1.05, -0.05), 0]}
                tickFormatter={asPercent}
                tick={tick}
                stroke="var(--color-line-strong)"
                width={44}
              />
              <Tooltip
                content={<CurveTooltip />}
                cursor={{ stroke: "var(--color-line-strong)", strokeWidth: 1 }}
              />
              <Line
                type="monotone"
                dataKey="max_drawdown"
                stroke="var(--color-brick)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Panel>
    </div>
  );
}
