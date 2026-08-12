/** Policy limits, one row per asset class.
 *
 * A real investment policy statement sets bounds asset by asset -- "at least
 * 5% cash, no more than 20% alternatives" -- rather than picking out two or
 * three. Offering a floor and a cap for every bucket means the form can
 * express an actual mandate instead of an approximation of one.
 *
 * Blank means no limit, which is different from zero: a cap of 0% forbids the
 * asset entirely, while no cap allows anything. Keeping those distinct is why
 * the fields hold strings rather than numbers.
 */

import type { AssetMeta } from "@/lib/api";
import { Explainer, Input } from "@/components/ui/primitives";

export type Limits = Record<string, { floor: string; cap: string }>;

export const emptyLimits = (assets: string[]): Limits =>
  Object.fromEntries(assets.map((key) => [key, { floor: "", cap: "" }]));

export function PolicyLimits({
  assets,
  limits,
  onChange,
  groupCap,
  onGroupCapChange,
}: {
  assets: AssetMeta[];
  limits: Limits;
  onChange: (next: Limits) => void;
  groupCap: string;
  onGroupCapChange: (value: string) => void;
}) {
  const set = (key: string, side: "floor" | "cap", value: string) =>
    onChange({
      ...limits,
      [key]: { ...(limits[key] ?? { floor: "", cap: "" }), [side]: value },
    });

  return (
    <div>
      <div className="mb-2 grid grid-cols-[1fr_4.5rem_4.5rem] items-end gap-2">
        <span className="eyebrow">asset</span>
        <span className="eyebrow text-right">at least</span>
        <span className="eyebrow text-right">at most</span>
      </div>

      <div className="space-y-2">
        {assets.map((asset) => (
          <div
            key={asset.key}
            className="grid grid-cols-[1fr_4.5rem_4.5rem] items-center gap-2"
          >
            <span className="truncate text-[0.8125rem] text-ink" title={asset.proxy}>
              {asset.label}
            </span>
            <Input
              inputMode="decimal"
              placeholder="—"
              aria-label={`${asset.label} minimum weight, percent`}
              value={limits[asset.key]?.floor ?? ""}
              onChange={(e) => set(asset.key, "floor", e.target.value)}
              className="px-2 py-1.5 text-right text-xs"
            />
            <Input
              inputMode="decimal"
              placeholder="—"
              aria-label={`${asset.label} maximum weight, percent`}
              value={limits[asset.key]?.cap ?? ""}
              onChange={(e) => set(asset.key, "cap", e.target.value)}
              className="px-2 py-1.5 text-right text-xs"
            />
          </div>
        ))}
      </div>

      <div className="mt-4 border-t border-line pt-4">
        <div className="grid grid-cols-[1fr_4.5rem_4.5rem] items-center gap-2">
          <span className="flex items-center text-[0.8125rem] text-ink">
            Growth assets
            <Explainer title="growth assets">
              Public equity and private equity taken together. Mandates usually
              cap total exposure to a kind of risk rather than to a label —
              and since the private equity proxy correlates around 0.9 with
              public equity, holding both is closer to one bet than two.
            </Explainer>
          </span>
          <span />
          <Input
            inputMode="decimal"
            placeholder="—"
            aria-label="Growth assets maximum weight, percent"
            value={groupCap}
            onChange={(e) => onGroupCapChange(e.target.value)}
            className="px-2 py-1.5 text-right text-xs"
          />
        </div>
      </div>

      <p className="mt-3 text-xs text-muted">
        Percentages. Blank means no limit — which is not the same as 0%, and a
        cap of 0% excludes the asset.
      </p>
    </div>
  );
}
