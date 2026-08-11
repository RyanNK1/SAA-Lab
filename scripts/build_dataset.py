"""Build the committed dataset. Run locally, then commit the CSV.

    python scripts/build_dataset.py

Writes data/public/monthly_returns.csv -- monthly returns per asset class,
from public sources. That file is what every later phase reads. The deployed
application never fetches historical data, which removes the single largest
deployment fragility: the price source rate-limits cloud hosts, so a fetch that
works on a laptop can fail once deployed.

Read the validation report before committing the output. It is not decoration:
a bad splice or a stale series shows up there and nowhere else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.market_data import build_panel, validate  # noqa: E402
from core.panels import levels_correlation_bias  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "public" / "monthly_returns.csv"

# Sanity bounds for the validation review. Deliberately wide -- these catch
# construction errors, not unusual markets.
PLAUSIBLE_VOL = (0.001, 0.60)
PLAUSIBLE_MONTHLY_RETURN = (-0.60, 0.60)


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def review(report: pd.DataFrame) -> list[str]:
    """Flag anything that looks like a construction error rather than a market.

    Returns a list of warnings. These are advisory: an unusual market can
    legitimately trip them, so they prompt a look rather than blocking.
    """
    warnings: list[str] = []

    for asset in report.index:
        vol = report.loc[asset, "ann_vol"]
        if not PLAUSIBLE_VOL[0] <= vol <= PLAUSIBLE_VOL[1]:
            warnings.append(
                f"{asset}: annual volatility {vol:.2%} is outside the plausible "
                f"range {PLAUSIBLE_VOL[0]:.1%}-{PLAUSIBLE_VOL[1]:.0%}"
            )

        low = report.loc[asset, "min_return"]
        high = report.loc[asset, "max_return"]
        if low < PLAUSIBLE_MONTHLY_RETURN[0]:
            warnings.append(
                f"{asset}: worst month {low:.2%}. A single catastrophic month "
                f"is the signature of a splice done wrong -- check the junction."
            )
        if high > PLAUSIBLE_MONTHLY_RETURN[1]:
            warnings.append(
                f"{asset}: best month {high:.2%}, implausibly large. Same check."
            )

        stale = report.loc[asset, "zero_return_months"]
        n = report.loc[asset, "n_obs"]
        if stale > max(3, 0.05 * n):
            warnings.append(
                f"{asset}: {stale} months with exactly zero return out of {n}. "
                f"Suggests repeated quotes rather than real flat months."
            )

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", default="2000-01-01", help="earliest date to request"
    )
    parser.add_argument("--end", default=None, help="latest date to request")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and report, but do not write the CSV",
    )
    args = parser.parse_args()

    history = build_panel(start=args.start, end=args.end)
    panel = history.returns()

    rule("VALIDATION REPORT")
    report = validate(history)
    print(
        report.to_string(
            formatters={
                "min_return": lambda x: f"{x:8.2%}",
                "max_return": lambda x: f"{x:8.2%}",
                "ann_return": lambda x: f"{x:8.2%}",
                "ann_vol": lambda x: f"{x:8.2%}",
            }
        )
    )

    warnings = review(report)
    if warnings:
        print(f"\n{len(warnings)} thing(s) to check:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("\nNothing flagged: volatilities, extremes and staleness all "
              "look plausible.")

    rule("COVERAGE")
    print(f"{len(panel)} monthly returns")
    print(f"{panel.start:%Y-%m} to {panel.end:%Y-%m}")
    print(f"assets: {', '.join(panel.assets)}")

    rule("CORRELATION MATRIX (monthly returns)")
    corr = panel.corr()
    print(corr.to_string(float_format=lambda x: f"{x: .3f}"))

    off_diagonal = corr.where(~pd.DataFrame(
        [[i == j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns,
    ))
    highest = off_diagonal.stack().idxmax()
    lowest = off_diagonal.stack().idxmin()
    print(
        f"\nMost correlated:  {highest[0]} / {highest[1]} "
        f"({off_diagonal.loc[highest]: .3f})"
    )
    print(
        f"Least correlated: {lowest[0]} / {lowest[1]} "
        f"({off_diagonal.loc[lowest]: .3f})"
    )

    rule("LEVELS VERSUS RETURNS")
    print(
        "The same pairs computed on price levels instead of returns. The "
        "levels\ncolumn is an artefact of shared trend -- any two series that "
        "rose over the\nperiod correlate near 1.0 whether or not their "
        "movements relate. Shown in\nfull as a standing check that the "
        "distinction above is being maintained."
    )
    print()
    print(
        levels_correlation_bias(history).to_string(
            index=False, float_format=lambda x: f"{x: .3f}"
        )
    )

    if args.dry_run:
        print("\n[dry run] nothing written")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = panel.returns.copy()
    out.insert(0, "currency", "USD")
    out.to_csv(OUT, float_format="%.10f")

    print(f"\nwrote {OUT}")
    print(f"  {len(out)} rows x {len(out.columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
