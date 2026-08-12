/** The envelope.
 *
 * This is the one element the interface is built around, because it carries
 * the argument the whole tool exists to make: a mandate is not met by one
 * allocation but by a space of them, and reporting a single set of weights
 * implies a precision the data does not support.
 *
 * Each row is one asset. The bar spans the range that asset's weight takes
 * across every qualifying allocation; the marker is where the best-ranked one
 * sits within it. A bar running the full width means the mandate has no
 * opinion about that asset at all -- anything works. A narrow bar means the
 * mandate is genuinely pinning it down, which is the more interesting and much
 * rarer case.
 *
 * The rows are ordered by how wide the range is, so what the mandate does not
 * determine comes first. That ordering is the point: it puts the least
 * decided thing at the top rather than burying it.
 */

import type { Allocation, EnvelopeRow } from "@/lib/api";
import { assetLabel, pct } from "@/lib/format";

const GOLD_SLEEVE = "commodities";

export function Envelope({
  rows,
  best,
}: {
  rows: EnvelopeRow[];
  best?: Allocation;
}) {
  const ordered = [...rows].sort((a, b) => b.spread - a.spread);
  const widest = Math.max(...ordered.map((r) => r.max), 0.01);

  return (
    <div>
      <div className="mb-4 flex items-baseline justify-between">
        <p className="max-w-lg text-sm leading-relaxed text-muted">
          Every allocation in these ranges meets the mandate. A wide bar is an
          asset the mandate leaves open; a narrow one is an asset it decides.
        </p>
        <span className="eyebrow shrink-0">weight</span>
      </div>

      <div className="space-y-3.5">
        {ordered.map((row) => {
          const left = (row.min / widest) * 100;
          const width = Math.max(((row.max - row.min) / widest) * 100, 0.4);
          const marker =
            best && typeof best[row.asset] === "number"
              ? ((best[row.asset] as number) / widest) * 100
              : null;

          return (
            <div key={row.asset} className="grid grid-cols-[9.5rem_1fr_5rem] items-center gap-4">
              <span className="truncate text-[0.8125rem] text-ink">
                {assetLabel(row.asset)}
              </span>

              <div className="relative h-6">
                <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-line" />
                <div
                  className="absolute top-1/2 h-2 -translate-y-1/2"
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                    background:
                      row.asset === GOLD_SLEEVE
                        ? "var(--color-brass)"
                        : "var(--color-primary)",
                    opacity: 0.28,
                  }}
                />
                {marker !== null && (
                  <div
                    className="absolute top-1/2 h-5 w-0.5 -translate-x-1/2 -translate-y-1/2"
                    style={{
                      left: `${marker}%`,
                      background:
                        row.asset === GOLD_SLEEVE
                          ? "var(--color-brass)"
                          : "var(--color-primary)",
                    }}
                    title={`Best-ranked allocation: ${pct(best?.[row.asset] as number)}`}
                  />
                )}
              </div>

              <span className="tabular text-right text-xs text-muted">
                {pct(row.min, 0)}&ndash;{pct(row.max, 0)}
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-5 border-t border-line pt-3 text-xs text-muted">
        The tick marks the allocation ranked first below. It is one member of
        the range, not the answer.
      </p>
    </div>
  );
}
