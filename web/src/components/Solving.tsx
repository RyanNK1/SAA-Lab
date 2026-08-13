/** What the interface shows while a mandate is being solved.
 *
 * The API answers in one blocking request and reports nothing along the way,
 * so there is no true progress to display. A bar that pretended otherwise
 * would be inventing information -- the familiar one that races to 90% and
 * then sits there.
 *
 * What can be shown honestly is elapsed time against an estimate, and the
 * estimate is stated as one. If the solve runs past it the bar stops advancing
 * and says so, rather than continuing to imply it knows how much is left.
 */

import { useEffect, useState } from "react";
import { Panel } from "@/components/ui/primitives";

/** Roughly how long a solve takes, in seconds.
 *
 * Monthly rebalancing is measured in one matrix operation across every sampled
 * allocation at once. Any other schedule has to simulate each allocation
 * individually, because the weights drift between trades and the path differs
 * -- about two orders of magnitude slower per allocation.
 *
 * These constants are measured, not guessed, and they are approximate on
 * purpose: the display treats them as an estimate rather than a promise.
 */
export function estimateSeconds(samples: number, schedule: string): number {
  const perSample = schedule === "monthly" ? 0.0004 : 0.015;
  return Math.max(1, samples * perSample);
}

export function Solving({
  samples,
  schedule,
}: {
  samples: number;
  schedule: string;
}) {
  const [elapsed, setElapsed] = useState(0);
  const estimate = estimateSeconds(samples, schedule);

  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(
      () => setElapsed((Date.now() - started) / 1000),
      100,
    );
    return () => window.clearInterval(timer);
  }, []);

  const overrun = elapsed > estimate;
  // Held at 92% rather than run to the end: the solve is not finished, and a
  // full bar would say it was.
  const filled = Math.min((elapsed / estimate) * 92, 92);

  return (
    <Panel title="solving">
      <div className="flex items-baseline justify-between gap-4">
        <p className="text-[0.9375rem] text-ink">
          Measuring {samples.toLocaleString()} allocations over the period.
        </p>
        <span className="tabular shrink-0 text-sm text-muted">
          {elapsed.toFixed(1)}s
        </span>
      </div>

      <div
        className="mt-4 h-1 w-full overflow-hidden bg-line"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(filled)}
        aria-label="Solving the mandate"
      >
        <div
          className="h-full bg-primary transition-[width] duration-200 ease-linear"
          style={{ width: `${filled}%` }}
        />
      </div>

      <p className="mt-3 text-xs leading-relaxed text-muted">
        {overrun
          ? "Taking longer than the estimate. Still working — the bar has stopped because there is nothing left to base it on."
          : `Usually about ${Math.round(estimate)}s at this setting. The bar tracks elapsed time against that estimate, not actual progress.`}
      </p>

      {schedule !== "monthly" && (
        <p className="mt-3 border-t border-line pt-3 text-xs leading-relaxed text-muted">
          <span className="text-ink">{schedule}</span> rebalancing is slow to
          measure: each allocation has to be simulated on its own, because the
          weights drift between trades. Monthly is measured in a single pass.
        </p>
      )}
    </Panel>
  );
}
