/** Formatting for figures shown in the interface.
 *
 * Percentages are fractions everywhere -- 0.06 for 6% -- matching the API.
 * Converting for display in one place stops the two representations getting
 * mixed up inside components.
 */

export const pct = (value: number | null | undefined, places = 1): string =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "--"
    : `${(value * 100).toFixed(places)}%`;

export const num = (value: number | null | undefined, places = 3): string =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "--"
    : value.toFixed(places);

/** Months, or "never" -- how the API reports a drawdown the portfolio did not
 *  recover from. It arrives as null, because infinity is not valid JSON. */
export const months = (value: number | null | undefined): string =>
  value === null || value === undefined ? "never" : `${Math.round(value)}`;

export const dateLabel = (iso: string): string =>
  new Date(iso).toLocaleDateString("en-GB", { year: "numeric", month: "short" });

export const assetLabel = (key: string): string =>
  key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
