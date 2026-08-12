"""Solve a mandate on the real dataset.

    python scripts/run_mandate.py
    python scripts/run_mandate.py --target 0.07 --max-vol 0.08
    python scripts/run_mandate.py --rebalance annual --cost-bps 10
    python scripts/run_mandate.py --across-periods
    python scripts/run_mandate.py --sweep

A mandate is the instruction an investor actually receives: what must be
achieved, what may not be exceeded, and what rules bind the allocation. The
question is not "which allocation scores highest" but "which allocations
satisfy all of this, and is there even one".

When several qualify -- which is the usual case -- they are all reported and
ranked on request. Which one is best depends on what the holder cares about,
and that is not a question the code can settle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Rebalance  # noqa: E402
from core.constraints import Constraints, GroupLimit  # noqa: E402
from core.mandate import (  # noqa: E402
    RANKABLE,
    Mandate,
    frontier_of_mandates,
    solve_mandate,
    solve_mandate_across_periods,
)
from core.optimize import DEFAULT_SAMPLES  # noqa: E402
from core.panels import ReturnPanel  # noqa: E402
from core.periods import resolve_periods  # noqa: E402
from core.rebalance import RebalanceSpec  # noqa: E402
from core.sleeve import build_sleeve  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "monthly_returns.csv"

GROWTH = ("equity", "private_equity")


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=0.06, help="minimum return")
    parser.add_argument("--max-vol", type=float, default=0.10, help="volatility budget")
    parser.add_argument("--max-dd", type=float, default=None, help="e.g. -0.25")
    parser.add_argument("--max-recovery", type=int, default=None, help="months")
    parser.add_argument("--cash-floor", type=float, default=0.05)
    parser.add_argument("--pe-cap", type=float, default=0.20)
    parser.add_argument("--growth-cap", type=float, default=0.60)
    parser.add_argument(
        "--rebalance", default="monthly", choices=[r.value for r in Rebalance]
    )
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--gold-weight", type=float, default=0.5)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--rank-by", default="max_drawdown", choices=sorted(RANKABLE))
    parser.add_argument(
        "--across-periods",
        action="store_true",
        help="require the mandate to be met in every regime, not just overall",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="find where the target stops being reachable"
    )
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

    constraints = Constraints(
        caps={"private_equity": args.pe_cap},
        floors={"cash": args.cash_floor},
        groups=(GroupLimit("growth", GROWTH, maximum=args.growth_cap),),
    )
    mandate = Mandate(
        target_return=args.target,
        max_volatility=args.max_vol,
        max_drawdown=args.max_dd,
        max_recovery_months=args.max_recovery,
        constraints=constraints,
        rebalance=RebalanceSpec(Rebalance(args.rebalance), args.cost_bps),
    )

    print(f"{len(sleeved)} monthly returns, {sleeved.start:%Y-%m} to {sleeved.end:%Y-%m}")
    print(f"sleeve:   {args.gold_weight:.0%} gold / {1 - args.gold_weight:.0%} ex-gold")
    print(f"samples:  {args.samples:,}")
    print(f"\nMANDATE:  {mandate.describe()}")

    if args.sweep:
        rule("HOW MUCH COULD HAVE BEEN ASKED FOR")
        print(f"Sweeping the return target at a {args.max_vol:.0%} volatility")
        print("budget, to find where the mandate stops being achievable.\n")
        targets = [round(t, 3) for t in np.arange(0.02, 0.13, 0.01)]
        frontier = frontier_of_mandates(
            sleeved,
            targets,
            args.max_vol,
            constraints,
            risk_free=cash,
            n_samples=args.samples,
        )
        print(
            frontier.to_string(
                index=False,
                formatters={
                    "target": lambda x: f"{x:7.1%}",
                    "n_qualifying": lambda x: f"{int(x):12d}",
                    "best_drawdown": lambda x: "" if pd.isna(x) else f"{x:13.2%}",
                    "return_at_best": lambda x: "" if pd.isna(x) else f"{x:14.2%}",
                    "vol_at_best": lambda x: "" if pd.isna(x) else f"{x:11.2%}",
                },
                na_rep="--",
            )
        )
        reachable = frontier[frontier["feasible"]]
        if len(reachable):
            print(f"\nHighest reachable target: {reachable['target'].max():.1%}")
        else:
            print("\nNo target in the swept range was reachable.")
        return 0

    if args.across_periods:
        periods = resolve_periods(sleeved)
        rule("MEETING THE MANDATE IN EVERY REGIME")
        print("Meeting a mandate over one window is a hindsight result -- the")
        print("allocation was chosen knowing what happened. Meeting it in the")
        print("crisis and the recovery and the inflation shock is the harder")
        print("and more meaningful claim.\n")

        robust = solve_mandate_across_periods(
            sleeved, mandate, periods, n_samples=args.samples
        )
        print(robust.explain())
        print()
        counts = robust.survival_counts()
        print(
            pd.DataFrame({"qualified": counts}).to_string(
                formatters={"qualified": lambda x: f"{int(x):9,d}"}
            )
        )

        if robust.any_survivors:
            print("\nWeight ranges across the allocations that survived everywhere:")
            envelope = pd.DataFrame(
                {
                    "min": robust.survivors[sleeved.assets].min(),
                    "median": robust.survivors[sleeved.assets].median(),
                    "max": robust.survivors[sleeved.assets].max(),
                }
            )
            print(envelope.to_string(float_format=lambda x: f"{x:8.1%}"))
        return 0

    result = solve_mandate(sleeved, mandate, risk_free=cash, n_samples=args.samples)

    rule("CAN IT BE DONE?")
    print(result.explain())

    if not result.feasible:
        return 0

    rule(f"QUALIFYING ALLOCATIONS, RANKED BY {args.rank_by.upper()}")
    print("All of these meet the mandate. Which is best depends on what you")
    print("care about -- rank by any column with --rank-by.\n")
    columns = sleeved.assets + [
        "realised_return",
        "volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "months_to_recover",
        "months_underwater",
    ]
    top = result.ranked(args.rank_by, limit=10)[columns]
    print(
        top.to_string(
            index=False,
            formatters={
                **{a: (lambda x: f"{x:7.1%}") for a in sleeved.assets},
                "realised_return": lambda x: f"{x:8.2%}",
                "volatility": lambda x: f"{x:8.2%}",
                "sharpe": lambda x: f"{x:7.3f}",
                "sortino": lambda x: f"{x:8.3f}",
                "max_drawdown": lambda x: f"{x:9.2%}",
                "months_to_recover": lambda x: "never" if np.isinf(x) else f"{int(x):5d}",
                "months_underwater": lambda x: f"{int(x):6d}",
            },
        )
    )

    rule("THE SHAPE OF THE ANSWER")
    print("Not one allocation but a space of them. An asset with a wide range")
    print("is one the mandate has no opinion about.\n")
    print(result.envelope().to_string(float_format=lambda x: f"{x:8.1%}"))

    rule("HOW MUCH ROOM TO SPARE")
    print("An allocation that only just clears the budget is a different")
    print("proposition from one comfortably inside it.\n")
    headroom = result.headroom()
    best = result.ranked(args.rank_by).index[0]
    print(f"best by {args.rank_by}:")
    print(headroom.loc[best].to_string(float_format=lambda x: f"{x:8.2%}"))
    print("\nacross all qualifying allocations:")
    print(headroom.describe().loc[["min", "50%", "max"]].to_string(
        float_format=lambda x: f"{x:8.2%}"
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
