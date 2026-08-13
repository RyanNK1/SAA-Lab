/** Allocations the user has set aside to compare.
 *
 * Stored by their weights rather than by position in a table. Ranking by
 * drawdown and then by return reorders the same rows, so a selection held as
 * "#3" would silently start pointing at a different portfolio -- the ticks
 * would appear to persist while quietly following the wrong allocations.
 *
 * Holding them by value also means a selection survives re-solving the mandate
 * entirely. Someone can loosen a constraint, solve again, keep two candidates
 * from each run and compare all four through the regimes, which is the
 * comparison the second view exists for.
 */

import type { Allocation } from "@/lib/api";

export interface Saved {
  /** Stable identity, derived from the weights themselves. */
  id: string;
  /** Where it came from, so a basket assembled over several solves is legible. */
  label: string;
  weights: Record<string, number>;
}

/** Weights rounded to a tenth of a percent, joined in a fixed order.
 *
 * Rounding matters: two allocations differing in the twelfth decimal are the
 * same portfolio for any purpose here, and treating them as distinct would let
 * the same row be added twice. */
export function identify(
  weights: Record<string, number>,
  assets: string[],
): string {
  return assets
    .map((asset) => `${asset}:${((weights[asset] ?? 0) * 1000).toFixed(0)}`)
    .join("|");
}

export function toSaved(
  allocation: Allocation,
  assets: string[],
  label: string,
): Saved {
  const weights = Object.fromEntries(
    assets.map((asset) => [asset, (allocation[asset] as number) ?? 0]),
  );
  return { id: identify(weights, assets), label, weights };
}

export function toggle(basket: Saved[], entry: Saved): Saved[] {
  return basket.some((s) => s.id === entry.id)
    ? basket.filter((s) => s.id !== entry.id)
    : [...basket, entry];
}

export const contains = (basket: Saved[], id: string): boolean =>
  basket.some((s) => s.id === id);

/** A short name for where a row came from: the measure it was ranked by and
 *  its position under that ranking. Two allocations picked from different
 *  rankings are then distinguishable in the basket. */
export function labelFor(rankBy: string, index: number): string {
  const short: Record<string, string> = {
    max_drawdown: "drawdown",
    realised_return: "return",
    volatility: "vol",
    sharpe: "sharpe",
    sortino: "sortino",
    months_to_recover: "recovery",
    months_underwater: "underwater",
  };
  return `${short[rankBy] ?? rankBy} #${index + 1}`;
}
