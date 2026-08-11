"""Compare rebalancing schedules on the real dataset.

    python scripts/run_rebalance.py
    python scripts/run_rebalance.py --cost-bps 25
    python scripts/run_rebalance.py --start 2007-01-01 --end 2012-12-31
    python scripts/run_rebalance.py --weights 60/10/20/7.5/2.5

A development script. It answers one question: does the rebalancing setting
change the outcome enough to be worth exposing, and in which direction?

No setting is recommended. Rebalancing is often sold as free money and is not:
it helps when assets oscillate and costs you when one genuinely outperforms
for a decade, because you kept trimming the winner. The output is meant to
show which of those happened over the chosen period.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Rebalance  # noqa: E402
from core.panels import ReturnPanel  # noqa: E402
from core.rebalance import (  # noqa: E402
    RebalanceSpec,
    measure_path,
    simulate_with_sleeve,
)
from core.sleeve import SLEEVE  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "monthly_returns.csv"

ORDER = ["equity", "fixed_income", "private_equity", SLEEVE, "cash"]
DEFAULT_WEIGHTS = (55.0, 12.5, 20.0, 7.5, 5.0)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def parse_weights(text: str) -> tuple[float, ...]:
    parts = tuple(float(p) for p in text.replace(",", "/").split("/"))
    if len(parts) != len(ORDER):
        raise ValueError(f"Expected {len(ORDER)} weights, got {len(parts)}")
    total = sum(parts)
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"Weights sum to {total:g}, not 100")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--gold-weight", type=float, default=0.5)
    parser.add_argument("--threshold-band", type=float, default=0.05)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--weights", default=None)
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

    percentages = parse_weights(args.weights) if args.weights else DEFAULT_WEIGHTS
    weights = {name: pct / 100.0 for name, pct in zip(ORDER, percentages)}

    print(f"{len(panel)} monthly returns, {panel.start:%Y-%m} to {panel.end:%Y-%m}")
    print(f"allocation:  {'/'.join(f'{p:g}' for p in percentages)}")
    print(f"             {' / '.join(ORDER)}")
    print(f"sleeve:      {args.gold_weight:.0%} gold / "
          f"{1 - args.gold_weight:.0%} ex-gold")
    print(f"costs:       {args.cost_bps:.0f}bps one-way on turnover")

    paths = {}
    rows = []
    for schedule in Rebalance:
        spec = RebalanceSpec(schedule, args.cost_bps, args.threshold_band)
        path = simulate_with_sleeve(panel, weights, args.gold_weight, spec)
        stats = measure_path(panel, path)
        paths[schedule.value] = (path, stats)

        rows.append(
            {
                "schedule": schedule.value,
                "return": stats.realised_return,
                "vol": stats.volatility,
                "sharpe": stats.sharpe,
                "sortino": stats.sortino,
                "max_dd": stats.max_drawdown,
                "trades": path.n_rebalances,
                "turnover": path.total_turnover,
                "cost": path.total_cost,
            }
        )

    rule("BY SCHEDULE")
    table = pd.DataFrame(rows).set_index("schedule")
    print(
        table.to_string(
            formatters={
                "return": lambda x: f"{x:7.2%}",
                "vol": lambda x: f"{x:7.2%}",
                "sharpe": lambda x: f"{x:6.3f}",
                "sortino": lambda x: f"{x:7.3f}",
                "max_dd": lambda x: f"{x:7.2%}",
                "trades": lambda x: f"{int(x):6d}",
                "turnover": lambda x: f"{x:8.2f}",
                "cost": lambda x: f"{x:7.2%}",
            }
        )
    )
    print("\n'turnover' is cumulative one-way turnover as a multiple of the")
    print("portfolio. 'cost' is the total paid over the period.")

    spread = table["return"].max() - table["return"].min()
    best_return = table["return"].idxmax()
    best_sharpe = table["sharpe"].idxmax()
    print(f"\nreturn spread across settings: {spread:.2%}")
    print(f"highest return: {best_return}   highest Sharpe: {best_sharpe}")
    if best_return != best_sharpe:
        print("These disagree: the higher-returning setting took more risk to")
        print("get there. Which is 'better' depends on what the holder wanted.")

    rule("HOW FAR THE PORTFOLIO WANDERED")
    print("Under 'never', nobody chose the ending allocation -- the winner grew")
    print("into it. That is the risk the setting exists to control.\n")
    never_path, _ = paths["never"]
    print(
        never_path.drift_summary().to_string(
            float_format=lambda x: f"{x: .1%}"
        )
    )

    rule("WORST DRAWDOWN")
    for name, (path, stats) in paths.items():
        print(f"  {name:<11} {stats.drawdown.describe()}")

    rule("WHAT THE TRADING COST")
    for name, (path, stats) in paths.items():
        gross = stats.realised_return + path.total_cost / (len(panel) / 12)
        print(
            f"  {name:<11} {path.n_rebalances:4d} trades, "
            f"{path.total_cost:6.2%} paid, "
            f"roughly {path.total_cost / (len(panel) / 12) * 10_000:5.1f}bps a year"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
