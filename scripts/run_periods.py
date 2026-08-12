"""Compare periods on the real dataset.

    python scripts/run_periods.py
    python scripts/run_periods.py --rolling 5
    python scripts/run_periods.py --objective min_drawdown
    python scripts/run_periods.py --cash-floor 0.05 --pe-cap 0.20

One period is a lookup. Several side by side is an argument.

Every answer here is hindsight -- what would have been best over each window,
chosen knowing what happened in it. The cross-period matrix is the honest test
of that: an allocation optimised for the crisis will look superb in the crisis
and may look terrible everywhere else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Objective  # noqa: E402
from core.constraints import Constraints  # noqa: E402
from core.optimize import DEFAULT_SAMPLES  # noqa: E402
from core.panels import ReturnPanel  # noqa: E402
from core.periods import (  # noqa: E402
    compare_periods,
    consensus_allocation,
    cross_period_performance,
    hindsight_premium,
    resolve_periods,
    rolling_periods,
    weight_stability,
)
from core.portfolio import portfolio_stats  # noqa: E402
from core.sleeve import build_sleeve  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "monthly_returns.csv"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective", default="max_sharpe", choices=[o.value for o in Objective]
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=None,
        help="use rolling windows of N years instead of named regimes",
    )
    parser.add_argument("--gold-weight", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--cash-floor", type=float, default=None)
    parser.add_argument("--pe-cap", type=float, default=None)
    args = parser.parse_args()

    if not DATA.exists():
        print(f"Missing {DATA}. Run scripts/build_dataset.py first.")
        return 1

    panel = ReturnPanel(
        pd.read_csv(DATA, index_col=0, parse_dates=True).drop(
            columns=["currency"], errors="ignore"
        )
    )
    sleeved = build_sleeve(panel, args.gold_weight)
    objective = Objective(args.objective)

    constraints = Constraints(
        caps={} if args.pe_cap is None else {"private_equity": args.pe_cap},
        floors={} if args.cash_floor is None else {"cash": args.cash_floor},
    )
    constraints.validate(sleeved.assets)

    periods = (
        rolling_periods(sleeved, years=args.rolling)
        if args.rolling
        else resolve_periods(sleeved)
    )

    print(f"{len(sleeved)} monthly returns, {sleeved.start:%Y-%m} to {sleeved.end:%Y-%m}")
    print(f"objective:   {objective.value}")
    print(f"sleeve:      {args.gold_weight:.0%} gold / {1 - args.gold_weight:.0%} ex-gold")
    print(f"constraints: {constraints.describe()}")
    print(f"periods:     {len(periods)}")

    table, results = compare_periods(
        sleeved,
        periods,
        objective=objective,
        constraints=constraints,
        n_samples=args.samples,
    )

    rule("BEST ALLOCATION, BY PERIOD")
    columns = ["months"] + sleeved.assets + ["return", "vol", "sharpe", "max_dd"]
    print(
        table[columns].to_string(
            formatters={
                **{a: (lambda x: f"{x:7.1%}") for a in sleeved.assets},
                "months": lambda x: f"{int(x):6d}",
                "return": lambda x: f"{x:8.2%}",
                "vol": lambda x: f"{x:7.2%}",
                "sharpe": lambda x: f"{x:7.3f}",
                "max_dd": lambda x: f"{x:8.2%}",
            }
        )
    )

    rule("HOW MUCH EACH ASSET'S ANSWER DEPENDS ON THE PERIOD")
    print("An asset whose optimal weight swings between windows is not")
    print("something the data has an opinion about. It is something the")
    print("period has an opinion about.\n")
    stability = weight_stability(table, sleeved.assets)
    print(stability.to_string(float_format=lambda x: f"{x:8.1%}"))

    most = stability.index[0]
    least = stability.index[-1]
    print(f"\nleast stable: {most} ({stability.loc[most, 'spread']:.0%} spread)")
    print(f"most stable:  {least} ({stability.loc[least, 'spread']:.0%} spread)")

    rule("EACH PERIOD'S WINNER, MEASURED EVERYWHERE ELSE")
    print("Rows: the period an allocation was chosen for.")
    print("Columns: the period it was then measured in.")
    print("The diagonal is in-sample and wins by construction -- it was picked")
    print("knowing what happened. Read across a row to see whether it")
    print("survived elsewhere.\n")
    matrix = cross_period_performance(sleeved, results, periods)
    print(matrix.to_string(float_format=lambda x: f"{x: 7.3f}"))

    rule("WHAT HINDSIGHT WAS WORTH")
    print("For each period: how the allocation chosen for it compares with")
    print("allocations chosen for other periods, measured in it. That gap is")
    print("the value of knowing the answer in advance, which nobody does.\n")
    premium = hindsight_premium(matrix)
    print(premium.to_string(float_format=lambda x: f"{x: 8.3f}"))
    print(f"\naverage hindsight premium: {premium['premium'].mean():.3f} Sharpe")

    rule("THE ALLOCATION NO PERIOD VOTED AGAINST")
    print("Period-length-weighted average of the answers above. Not an optimum")
    print("for anything, and not presented as one -- a starting point for when")
    print("the periods disagree, which they usually do.\n")
    consensus = consensus_allocation(table, sleeved.assets)
    print(consensus.to_string(float_format=lambda x: f"{x:7.1%}"))

    cash = sleeved.returns["cash"] if "cash" in sleeved.assets else None
    whole = portfolio_stats(sleeved, consensus, risk_free=cash)
    print(f"\nover the full sample: return {whole.realised_return:.2%}, "
          f"vol {whole.volatility:.2%}, sharpe {whole.sharpe:.3f}, "
          f"drawdown {whole.max_drawdown:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
