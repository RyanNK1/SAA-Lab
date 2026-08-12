"""Comparing periods.

A single period is a lookup. Several side by side is an argument.

The best allocation for 2008-2012 looks nothing like the one for 2013-2019,
and the *way* it changes is the insight: gold dominates one window and is dead
weight in another; fixed income diversifies equities for fifteen years and then
stops in 2022. None of that is visible from one answer averaged across
everything.

This module runs the same question -- same objective, same constraints, same
sleeve setting -- across several windows, and reports where the answers agree
and where they diverge. Agreement across regimes is the closest this tool comes
to a durable conclusion. Disagreement is the more common and more honest
result, and it is what tells a user how much any single answer is worth.

Two framings are supported:

  **Named regimes.** Pre-defined windows with labels -- the crisis, the
  zero-rate decade, the inflation shock. Useful to someone who does not know
  the history well enough to pick dates.

  **Rolling windows.** Every N-year window, stepped through the sample. Useful
  for seeing whether a conclusion is stable or an artefact of one stretch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import Objective
from core.constraints import Constraints, optimize_constrained
from core.optimize import DEFAULT_SAMPLES, DEFAULT_TOLERANCE, OptimizationResult
from core.panels import ReturnPanel
from core.portfolio import portfolio_stats

# Windows chosen for how the world behaved, not for round numbers. Each is a
# genuinely different environment for a multi-asset portfolio.
NAMED_REGIMES: tuple[tuple[str, str, str], ...] = (
    ("Run-up", "2006-01-01", "2007-09-30"),
    ("Crisis", "2007-10-01", "2009-02-28"),
    ("Recovery", "2009-03-01", "2012-12-31"),
    ("Zero-rate decade", "2013-01-01", "2019-12-31"),
    ("Pandemic", "2020-01-01", "2021-12-31"),
    ("Inflation shock", "2022-01-01", "2023-12-31"),
    ("Recent", "2024-01-01", "2030-12-31"),
)


@dataclass(frozen=True)
class Period:
    """One named window."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def years(self) -> float:
        return (self.end - self.start).days / 365.25

    def describe(self) -> str:
        return f"{self.label} ({self.start:%Y-%m} to {self.end:%Y-%m})"


def resolve_periods(
    panel: ReturnPanel,
    periods: tuple[tuple[str, str, str], ...] = NAMED_REGIMES,
    min_observations: int = 12,
) -> list[Period]:
    """Clip named windows to what the data actually covers.

    Windows falling wholly outside the sample, or leaving too few observations
    to say anything, are dropped rather than reported with a handful of months
    behind them. A regime represented by four data points is not a regime.
    """
    resolved = []
    for label, start, end in periods:
        window_start = max(pd.Timestamp(start), panel.start)
        window_end = min(pd.Timestamp(end), panel.end)
        if window_start >= window_end:
            continue

        count = len(panel.returns.loc[window_start:window_end])
        if count < min_observations:
            continue

        resolved.append(Period(label, window_start, window_end))
    return resolved


def rolling_periods(
    panel: ReturnPanel, years: int = 5, step_months: int = 12
) -> list[Period]:
    """Overlapping windows of a fixed length, stepped through the sample.

    Overlapping windows are not independent evidence -- adjacent ones share
    most of their data. They answer a narrower question: whether a conclusion
    holds throughout the sample or only in one stretch of it.
    """
    if years < 1:
        raise ValueError("years must be at least 1")
    if step_months < 1:
        raise ValueError("step_months must be at least 1")

    window = pd.DateOffset(years=years)
    periods = []
    cursor = panel.start

    while cursor + window <= panel.end:
        end = cursor + window
        periods.append(
            Period(f"{cursor:%Y}-{end:%Y}", pd.Timestamp(cursor), pd.Timestamp(end))
        )
        cursor = cursor + pd.DateOffset(months=step_months)

    return periods


def compare_periods(
    panel: ReturnPanel,
    periods: list[Period] | None = None,
    objective: Objective = Objective.MAX_SHARPE,
    constraints: Constraints | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, OptimizationResult]]:
    """Solve the same problem in each window.

    Returns a table with one row per period -- the optimal weights and the
    statistics they achieved -- and the full results keyed by label.
    """
    periods = periods if periods is not None else resolve_periods(panel)
    if not periods:
        raise ValueError("No periods overlap the panel's date range")

    constraints = constraints or Constraints()
    constraints.validate(panel.assets)

    rows = []
    results: dict[str, OptimizationResult] = {}

    for period in periods:
        window = panel.between(period.start, period.end)
        cash = window.returns["cash"] if "cash" in window.assets else None

        result = optimize_constrained(
            window,
            objective,
            constraints,
            risk_free=cash,
            tolerance=tolerance,
            n_samples=n_samples,
            seed=seed,
        )
        results[period.label] = result

        row = {
            "period": period.label,
            "start": period.start,
            "end": period.end,
            "months": len(window),
        }
        row.update({asset: result.weights[asset] for asset in panel.assets})
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

    return pd.DataFrame(rows).set_index("period"), results


def weight_stability(table: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """How much each asset's optimal weight moves between periods.

    The central output. An asset whose weight ranges from 0% to 80% depending
    on the window is not something the data has an opinion about -- it is
    something the *period* has an opinion about. An asset that stays near the
    same weight throughout is the closest thing here to a durable conclusion.
    """
    weights = table[assets]
    return pd.DataFrame(
        {
            "mean": weights.mean(),
            "min": weights.min(),
            "max": weights.max(),
            "spread": weights.max() - weights.min(),
            "std": weights.std(ddof=1) if len(weights) > 1 else 0.0,
        }
    ).sort_values("spread", ascending=False)


def cross_period_performance(
    panel: ReturnPanel,
    results: dict[str, OptimizationResult],
    periods: list[Period],
    risk_free: pd.Series | None = None,
) -> pd.DataFrame:
    """How each period's winning allocation performed in every other period.

    The honest test of a hindsight answer. An allocation optimised for the
    crisis will look superb in the crisis -- it was chosen knowing what
    happened. Whether it survives elsewhere is the question that matters, and
    it is the one a single-period result cannot ask.

    Rows are the period an allocation was chosen for; columns are the period it
    was then measured in. The diagonal is in-sample by construction and should
    be ignored when reading down a column.
    """
    labels = [p.label for p in periods]
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for chosen_in, result in results.items():
        for period in periods:
            window = panel.between(period.start, period.end)
            cash = window.returns["cash"] if "cash" in window.assets else None
            stats = portfolio_stats(window, result.weights, risk_free=cash)
            matrix.loc[chosen_in, period.label] = stats.sharpe

    return matrix


def hindsight_premium(matrix: pd.DataFrame) -> pd.DataFrame:
    """How much better each period's winner did in-sample than others did there.

    For each period, the gap between the allocation chosen *for* that period
    and the average of allocations chosen for other periods, measured in that
    period. That gap is the value of having known the answer in advance -- and
    since nobody does, it is the amount by which any single-period result
    flatters itself.
    """
    rows = []
    for label in matrix.columns:
        column = matrix[label]
        in_sample = column.loc[label]
        others = column.drop(index=label)
        rows.append(
            {
                "period": label,
                "chosen_for_it": in_sample,
                "others_average": others.mean(),
                "best_other": others.max(),
                "premium": in_sample - others.mean(),
            }
        )
    return pd.DataFrame(rows).set_index("period")


def consensus_allocation(
    table: pd.DataFrame, assets: list[str], weight_by_months: bool = True
) -> pd.Series:
    """A single allocation averaged across periods.

    Not an optimum for anything, and it should not be presented as one. It is
    the allocation nobody's period voted against -- a reasonable starting point
    when the periods disagree, which they usually do.

    Weighting by length stops a three-month window counting as much as a
    seven-year one.
    """
    weights = table[assets]
    if weight_by_months and "months" in table.columns:
        months = table["months"].to_numpy(dtype=float)
        averaged = (weights.to_numpy() * months[:, None]).sum(axis=0) / months.sum()
    else:
        averaged = weights.to_numpy().mean(axis=0)

    series = pd.Series(averaged, index=assets)
    total = series.sum()
    return series / total if total > 0 else series
