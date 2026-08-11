"""Measure a set of allocations on the real dataset.

    python scripts/run_portfolio.py
    python scripts/run_portfolio.py --gold-weight 0.8
    python scripts/run_portfolio.py --start 2007-01-01 --end 2012-12-31
    python scripts/run_portfolio.py --weights 60/20/10/5/5

A development script, not a product feature. Its job is to confirm the
measurement layer behaves sensibly on real data across structurally different
portfolios -- not to recommend any of them.

Allocation order is: equity / fixed income / private equity / commodities /
cash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.panels import ReturnPanel  # noqa: E402
from core.portfolio import portfolio_stats, risk_contributions  # noqa: E402
from core.sleeve import SLEEVE, build_sleeve, sleeve_sensitivity  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "monthly_returns.csv"

ORDER = ["equity", "fixed_income", "private_equity", SLEEVE, "cash"]

SAMPLES: list[tuple[float, ...]] = [
    (45.0, 25.0, 15.0, 7.5, 7.5),
    (50.0, 12.5, 25.0, 7.5, 5.0),
    (55.0, 12.5, 20.0, 7.5, 5.0),
    (55.0, 15.0, 20.0, 7.5, 2.5),
    (60.0, 10.0, 20.0, 7.5, 2.5),
]


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def parse_weights(text: str) -> tuple[float, ...]:
    """Parse '60/20/10/5/5' into percentages, checking they sum to 100."""
    parts = tuple(float(p) for p in text.replace(",", "/").split("/"))
    if len(parts) != len(ORDER):
        raise ValueError(
            f"Expected {len(ORDER)} weights ({'/'.join(ORDER)}), got {len(parts)}"
        )
    total = sum(parts)
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"Weights sum to {total:g}, not 100")
    return parts


def to_fractions(percentages: tuple[float, ...]) -> dict[str, float]:
    return {name: pct / 100.0 for name, pct in zip(ORDER, percentages)}


def label(percentages: tuple[float, ...]) -> str:
    return "/".join(f"{p:g}" for p in percentages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-weight",
        type=float,
        default=0.5,
        help="commodities slider: 1.0 is all gold, 0.0 all ex-gold",
    )
    parser.add_argument("--start", default=None, help="period start, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="period end, YYYY-MM-DD")
    parser.add_argument(
        "--weights",
        action="append",
        default=None,
        help="an allocation as 'e/fi/pe/comm/cash' percentages; repeatable",
    )
    args = parser.parse_args()

    if not DATA.exists():
        print(f"Missing {DATA}. Run scripts/build_dataset.py first.")
        return 1

    panel = ReturnPanel(pd.read_csv(DATA, index_col=0, parse_dates=True).drop(
        columns=["currency"], errors="ignore"
    ))

    if args.start or args.end:
        panel = panel.between(
            args.start or panel.start, args.end or panel.end
        )

    sleeved = build_sleeve(panel, args.gold_weight)
    cash = sleeved.returns["cash"] if "cash" in sleeved.assets else 0.0

    allocations = (
        [parse_weights(w) for w in args.weights] if args.weights else SAMPLES
    )

    print(f"{len(sleeved)} monthly returns, {sleeved.start:%Y-%m} to {sleeved.end:%Y-%m}")
    print(f"commodities sleeve: {args.gold_weight:.0%} gold / "
          f"{1 - args.gold_weight:.0%} ex-gold")
    print(f"allocation order:   {' / '.join(ORDER)}")

    rule("PER-ASSET")
    print(
        pd.DataFrame(
            {"ann_return": sleeved.ann_return(), "ann_vol": sleeved.ann_vol()}
        ).to_string(float_format=lambda x: f"{x:8.2%}")
    )

    rule("ALLOCATIONS")
    rows = []
    computed = {}
    for percentages in allocations:
        weights = to_fractions(percentages)
        stats = portfolio_stats(sleeved, weights, risk_free=cash)
        computed[label(percentages)] = stats
        rows.append(
            {
                "allocation": label(percentages),
                "return": stats.realised_return,
                "vol": stats.volatility,
                "sharpe": stats.sharpe,
                "sortino": stats.sortino,
                "max_dd": stats.max_drawdown,
                "recover": stats.drawdown.months_to_recover,
                "underwater": stats.drawdown.months_underwater,
            }
        )

    table = pd.DataFrame(rows).set_index("allocation")
    print(
        table.to_string(
            formatters={
                "return": lambda x: f"{x:7.2%}",
                "vol": lambda x: f"{x:7.2%}",
                "sharpe": lambda x: f"{x:6.3f}",
                "sortino": lambda x: f"{x:7.3f}",
                "max_dd": lambda x: f"{x:7.2%}",
                "recover": lambda x: "never" if pd.isna(x) else f"{int(x):d}",
                "underwater": lambda x: f"{int(x):d}",
            },
            na_rep="never",
        )
    )
    print("\n'recover' and 'underwater' are months. Sharpe and Sortino are")
    print("invariant to the order months occur in; max_dd and recovery are not.")

    rule("WORST DRAWDOWN, BY ALLOCATION")
    for name, stats in computed.items():
        print(f"  {name:<22} {stats.drawdown.describe()}")

    rule("RISK CONTRIBUTIONS")
    print("Weight is not risk. An asset that is volatile and moves with the")
    print("rest of the portfolio carries more risk than its weight suggests.\n")
    for name, stats in computed.items():
        rc = risk_contributions(sleeved, stats.weights)
        summary = "  ".join(
            f"{asset}={rc.loc[asset, 'pct_of_risk']:.0%}" for asset in ORDER
        )
        print(f"  {name:<22} {summary}")

    rule("COMMODITIES SLIDER")
    print("The sleeve is not one asset. Its return, risk and every correlation")
    print("depend on where the slider sits.\n")
    if {"gold", "commodities_ex_gold"} <= set(panel.assets):
        sensitivity = sleeve_sensitivity(panel, steps=6)
        print(
            sensitivity.to_string(
                index=False,
                float_format=lambda x: f"{x: .3f}",
                formatters={
                    "gold_weight": lambda x: f"{x:11.0%}",
                    "ann_return": lambda x: f"{x:10.2%}",
                    "ann_vol": lambda x: f"{x:8.2%}",
                },
            )
        )
    else:
        print("  (both sleeve components not present in this panel)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
