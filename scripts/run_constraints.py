"""What policy constraints cost, on the real dataset.

    python scripts/run_constraints.py
    python scripts/run_constraints.py --objective min_drawdown
    python scripts/run_constraints.py --start 2007-01-01 --end 2012-12-31
    python scripts/run_constraints.py --pe-cap 0.10 --cash-floor 0.10

Solves the allocation twice -- once free, once under the limits -- and reports
the gap. The cost is always non-negative: a constraint shrinks the set of
allowed allocations, and a smaller set cannot contain a better answer.

Every answer is hindsight: what the rules cost over the window chosen. That is
exactly answerable and is not a forecast of what they will cost next.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Objective  # noqa: E402
from core.constraints import (  # noqa: E402
    Constraints,
    GroupLimit,
    cost_of_constraints,
    cost_per_constraint,
    optimize_constrained,
)
from core.optimize import DEFAULT_SAMPLES  # noqa: E402
from core.panels import ReturnPanel  # noqa: E402
from core.sleeve import build_sleeve  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "monthly_returns.csv"

GROWTH = ("equity", "private_equity")


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective",
        default="max_sharpe",
        choices=[o.value for o in Objective],
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--gold-weight", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--pe-cap", type=float, default=0.20)
    parser.add_argument("--cash-floor", type=float, default=0.05)
    parser.add_argument("--fi-floor", type=float, default=0.15)
    parser.add_argument("--growth-cap", type=float, default=0.60)
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
    objective = Objective(args.objective)

    constraints = Constraints(
        caps={"private_equity": args.pe_cap},
        floors={"cash": args.cash_floor, "fixed_income": args.fi_floor},
        groups=(GroupLimit("growth", GROWTH, maximum=args.growth_cap),),
    )
    constraints.validate(sleeved.assets)

    print(f"{len(sleeved)} monthly returns, {sleeved.start:%Y-%m} to {sleeved.end:%Y-%m}")
    print(f"objective:   {objective.value}")
    print(f"sleeve:      {args.gold_weight:.0%} gold / {1 - args.gold_weight:.0%} ex-gold")
    print(f"constraints: {constraints.describe()}")
    print(f"             growth = {' + '.join(GROWTH)}")
    print(f"samples:     {args.samples:,}")

    result = cost_of_constraints(
        sleeved,
        objective,
        constraints,
        risk_free=cash,
        n_samples=args.samples,
    )

    rule("FREE VERSUS CONSTRAINED")
    comparison = pd.DataFrame(
        {
            "unconstrained": result.unconstrained.weights,
            "constrained": result.constrained.weights,
            "change": result.constrained.weights - result.unconstrained.weights,
        }
    )
    print(comparison.to_string(float_format=lambda x: f"{x:8.1%}"))

    print()
    stats = pd.DataFrame(
        {
            "unconstrained": {
                "return": result.unconstrained.stats.realised_return,
                "volatility": result.unconstrained.stats.volatility,
                "sharpe": result.unconstrained.stats.sharpe,
                "sortino": result.unconstrained.stats.sortino,
                "max_drawdown": result.unconstrained.stats.max_drawdown,
            },
            "constrained": {
                "return": result.constrained.stats.realised_return,
                "volatility": result.constrained.stats.volatility,
                "sharpe": result.constrained.stats.sharpe,
                "sortino": result.constrained.stats.sortino,
                "max_drawdown": result.constrained.stats.max_drawdown,
            },
        }
    )
    stats["difference"] = stats["constrained"] - stats["unconstrained"]
    print(stats.to_string(float_format=lambda x: f"{x:9.4f}"))

    rule("WHAT THE RULES COST")
    print(f"  {result.describe()}")
    print()
    print(f"  return given up:  {result.return_cost * 10_000:7.1f}bps a year")
    print(f"  {objective.value} given up: {result.cost:10.4f}")

    if abs(result.cost) < 1e-9:
        print()
        print("  Nothing. The unconstrained optimum already satisfied every")
        print("  rule, so the limits are not binding over this period.")

    rule("COST OF EACH RULE, IN ISOLATION")
    print("Each row removes one rule and re-solves, so the figure is that")
    print("rule's marginal cost given the others. These do not sum to the")
    print("total: constraints interact, and two rules can be individually")
    print("cheap and jointly expensive.\n")
    per_rule = cost_per_constraint(
        sleeved, objective, constraints, risk_free=cash, n_samples=args.samples
    )
    print(
        per_rule[["kind", "constraint", "cost_bps"]].to_string(
            index=False, formatters={"cost_bps": lambda x: f"{x:9.1f}"}
        )
    )

    rule("DOES A FLOOR FIX THE CASH PROBLEM?")
    print("Unconstrained, minimising drawdown puts everything in cash: correct")
    print("for the question asked, useless as an allocation. A floor on risky")
    print("assets forces a real choice.\n")

    risky = tuple(a for a in sleeved.assets if a != "cash")
    forced = Constraints(
        groups=(GroupLimit("risky assets", risky, minimum=0.60),)
    )
    free_dd = optimize_constrained(
        sleeved, Objective.MIN_DRAWDOWN, Constraints(),
        risk_free=cash, n_samples=args.samples,
    )
    bound_dd = optimize_constrained(
        sleeved, Objective.MIN_DRAWDOWN, forced,
        risk_free=cash, n_samples=args.samples,
    )

    print(
        pd.DataFrame(
            {"unconstrained": free_dd.weights, "risky >= 60%": bound_dd.weights}
        ).to_string(float_format=lambda x: f"{x:8.1%}")
    )
    print()
    print(f"  drawdown, unconstrained: {free_dd.stats.max_drawdown:7.2%}")
    print(f"  drawdown, risky >= 60%:  {bound_dd.stats.max_drawdown:7.2%}")
    print(f"  return,   unconstrained: {free_dd.stats.realised_return:7.2%}")
    print(f"  return,   risky >= 60%:  {bound_dd.stats.realised_return:7.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
