"""Find the best allocation over a period, on the real dataset.

    python scripts/run_optimize.py
    python scripts/run_optimize.py --start 2007-01-01 --end 2012-12-31
    python scripts/run_optimize.py --max-weight 0.4 --tolerance 0.05
    python scripts/run_optimize.py --gold-weight 0.8 --samples 50000

Every answer here is hindsight: what *would* have been best over the window
chosen. That is exactly answerable and genuinely useful, and it is not a
forecast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Objective  # noqa: E402
from core.optimize import (  # noqa: E402
    DEFAULT_SAMPLES,
    DEFAULT_TOLERANCE,
    Method,
    efficient_frontier,
    optimize_all,
)
from core.panels import ReturnPanel  # noqa: E402
from core.portfolio import risk_contributions  # noqa: E402
from core.sleeve import build_sleeve  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "monthly_returns.csv"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--gold-weight", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()

    if not DATA.exists():
        print(f"Missing {DATA}. Run scripts/build_dataset.py first.")
        return 1

    panel = ReturnPanel(
        pd.read_csv(DATA, index_col=0, parse_dates=True).drop(
            columns=["currency"], errors="ignore"
        )
    )
    if args.start or args.end:
        panel = panel.between(args.start or panel.start, args.end or panel.end)

    sleeved = build_sleeve(panel, args.gold_weight)
    cash = sleeved.returns["cash"] if "cash" in sleeved.assets else None

    print(f"{len(sleeved)} monthly returns, {sleeved.start:%Y-%m} to {sleeved.end:%Y-%m}")
    print(f"sleeve:      {args.gold_weight:.0%} gold / {1 - args.gold_weight:.0%} ex-gold")
    print(f"cap:         {args.max_weight:.0%} per asset")
    print(f"tolerance:   {args.tolerance:.0%} of the best value counts as equivalent")
    print(f"samples:     {args.samples:,}")

    rule("PER-ASSET")
    print(
        pd.DataFrame(
            {"ann_return": sleeved.ann_return(), "ann_vol": sleeved.ann_vol()}
        ).to_string(float_format=lambda x: f"{x:8.2%}")
    )

    results = optimize_all(
        sleeved,
        max_weight=args.max_weight,
        risk_free=cash,
        tolerance=args.tolerance,
        n_samples=args.samples,
    )

    rule("BEST ALLOCATION, BY OBJECTIVE")
    rows = []
    for objective, result in results.items():
        row = {"objective": objective.value, "method": result.method.value}
        row.update({a: result.weights[a] for a in sleeved.assets})
        row.update(
            {
                "return": result.stats.realised_return,
                "vol": result.stats.volatility,
                "sharpe": result.stats.sharpe,
                "sortino": result.stats.sortino,
                "max_dd": result.stats.max_drawdown,
            }
        )
        rows.append(row)

    table = pd.DataFrame(rows).set_index("objective")
    print(
        table.to_string(
            formatters={
                **{a: (lambda x: f"{x:6.1%}") for a in sleeved.assets},
                "return": lambda x: f"{x:7.2%}",
                "vol": lambda x: f"{x:7.2%}",
                "sharpe": lambda x: f"{x:6.3f}",
                "sortino": lambda x: f"{x:7.3f}",
                "max_dd": lambda x: f"{x:7.2%}",
            }
        )
    )
    print("\n'exact' answers are solved algebraically. 'sampled' answers are the")
    print("best of a large search -- strong candidates, not proven optima,")
    print("because drawdown and Sortino have no usable gradient to follow.")

    rule("HOW MUCH THE PRECISION IS WORTH")
    print("Allocations within tolerance of the best perform equivalently.")
    print("A single answer would imply precision the data does not support.\n")
    for objective, result in results.items():
        n = len(result.near_optimal)
        print(f"--- {objective.value}  ({n:,} equivalent allocations found)")
        if n <= 1:
            print("    only one allocation qualifies at this tolerance\n")
            continue
        print(
            result.ranges().to_string(
                float_format=lambda x: f"{x:7.1%}",
                columns=["best", "low", "high"],
            )
        )
        print()

    rule("WHERE THE RISK SITS")
    print("Weight is not risk.\n")
    for objective, result in results.items():
        rc = risk_contributions(sleeved, result.weights)
        summary = "  ".join(
            f"{a}={rc.loc[a, 'pct_of_risk']:.0%}" for a in sleeved.assets
        )
        print(f"  {objective.value:<15} {summary}")

    rule("THE CURVE")
    print("Lowest volatility reachable at each level of return. The user picks")
    print("where on this to sit rather than being handed one answer.\n")
    curve = efficient_frontier(
        sleeved, n_points=25, max_weight=args.max_weight, risk_free=cash
    )
    columns = ["expected_return", "volatility", "sharpe", "max_drawdown"] + sleeved.assets
    print(
        curve[columns]
        .iloc[::4]
        .to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
