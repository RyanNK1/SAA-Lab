"""Simulation of how an allocation is actually held over time.

An allocation is not a one-time decision. Between rebalances the weights drift
with performance: if equity returns 10% and bonds return nothing, the equity
share grows. Left alone for twenty years, a 60/40 portfolio can end up 75/25 --
and nobody decided that. The winner grew into a larger share of the pot, so
risk rose silently, always in the direction of whatever had been winning, and
peaked right before the winner turned.

Rebalancing corrects it: sell what rose, buy what fell, return to target. That
feels wrong -- every instinct says back the winner -- but it is the mechanical
consequence of holding a target, and it is the only way the portfolio keeps
the shape its owner chose.

This module produces the *path*. What it hands back is a return series, which
goes straight into `portfolio.portfolio_stats`, so drawdown, Sortino and
everything else are computed identically regardless of the setting. Nothing in
the measurement layer needed rewriting to support this.

One thing this module deliberately does not do is recommend a setting.
Rebalancing is often sold as free money. It is not: it helps when assets
oscillate and costs you when one genuinely outperforms for a decade, because
you kept trimming the winner. Both happen, and the output should let a user see
which one did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import (
    DEFAULT_COST_BPS,
    DEFAULT_THRESHOLD_BAND,
    MONTHS_PER_YEAR,
    Rebalance,
)
from core.panels import ReturnPanel
from core.portfolio import as_weights

# How many periods between calendar rebalances, per setting.
_CALENDAR_INTERVAL: dict[Rebalance, int] = {
    Rebalance.MONTHLY: 1,
    Rebalance.QUARTERLY: 3,
    Rebalance.ANNUAL: 12,
}


@dataclass(frozen=True)
class RebalanceSpec:
    """How and how often a portfolio is corrected back to its targets."""

    schedule: Rebalance = Rebalance.ANNUAL
    cost_bps: float = DEFAULT_COST_BPS
    threshold_band: float = DEFAULT_THRESHOLD_BAND

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, Rebalance):
            raise TypeError(
                f"schedule must be a Rebalance member, got {type(self.schedule)}"
            )
        if self.cost_bps < 0:
            raise ValueError(f"cost_bps cannot be negative, got {self.cost_bps}")
        if not 0.0 < self.threshold_band < 1.0:
            raise ValueError(
                f"threshold_band must be between 0 and 1, got {self.threshold_band}"
            )

    @property
    def interval(self) -> int | None:
        """Periods between calendar rebalances, or None if not calendar-based."""
        return _CALENDAR_INTERVAL.get(self.schedule)

    def describe(self) -> str:
        if self.schedule is Rebalance.NEVER:
            return "never rebalanced"
        if self.schedule is Rebalance.THRESHOLD:
            return (
                f"rebalanced when any asset drifts "
                f"{self.threshold_band:.0%} off target, {self.cost_bps:.0f}bps"
            )
        return f"{self.schedule.value} rebalancing, {self.cost_bps:.0f}bps"


@dataclass(frozen=True)
class RebalancePath:
    """The realised history of holding an allocation under one setting."""

    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    targets: pd.Series
    spec: RebalanceSpec

    @property
    def n_rebalances(self) -> int:
        return int((self.turnover > 0).sum())

    @property
    def total_turnover(self) -> float:
        return float(self.turnover.sum())

    @property
    def total_cost(self) -> float:
        """Total trading cost paid, as a fraction of portfolio value."""
        return float(self.costs.sum())

    @property
    def avg_turnover(self) -> float:
        """Mean turnover across periods where trading actually happened."""
        traded = self.turnover[self.turnover > 0]
        return float(traded.mean()) if len(traded) > 0 else 0.0

    @property
    def final_weights(self) -> pd.Series:
        """Weights at the end of the period, after the last period's drift.

        Under `never`, comparing these to `targets` shows how far the portfolio
        wandered from the allocation its owner actually chose.
        """
        return self.weights.iloc[-1]

    @property
    def max_drift(self) -> float:
        """Largest distance any asset ever reached from its target weight."""
        return float((self.weights - self.targets).abs().to_numpy().max())

    def drift_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "target": self.targets,
                "final": self.final_weights,
                "drift": self.final_weights - self.targets,
                "max_drift": (self.weights - self.targets).abs().max(),
            }
        )


def _should_rebalance(
    period: int, current: np.ndarray, target: np.ndarray, spec: RebalanceSpec
) -> bool:
    """Whether to trade back to target at the start of this period."""
    if spec.schedule is Rebalance.NEVER:
        return False

    if spec.schedule is Rebalance.THRESHOLD:
        return bool(np.abs(current - target).max() > spec.threshold_band)

    interval = spec.interval
    assert interval is not None  # every remaining schedule is calendar-based
    return period % interval == 0


def simulate(
    panel: ReturnPanel,
    weights: pd.Series | dict[str, float],
    spec: RebalanceSpec | Rebalance | None = None,
) -> RebalancePath:
    """Hold an allocation across the panel and record what happened.

    Each period:

      1. Decide whether to rebalance. If so, trade back to target and charge
         the cost. Turnover is measured against the *drifted* weights, not the
         stale targets -- a portfolio should not be charged for trades it never
         had to make.
      2. Earn the period's return on whatever is held.
      3. Let the weights drift with performance, ready for the next period.

    The first period always establishes the position and is never charged
    turnover: there is no prior portfolio to trade out of.

    Returns are net of costs, so the series can be measured directly.
    """
    if spec is None:
        spec = RebalanceSpec()
    elif isinstance(spec, Rebalance):
        spec = RebalanceSpec(schedule=spec)

    target = as_weights(weights, panel.assets)
    target_array = target.to_numpy()
    rets = panel.returns

    current = target_array.copy()
    out_returns: list[float] = []
    weight_rows: list[np.ndarray] = []
    turnovers: list[float] = []
    costs: list[float] = []

    for period in range(len(rets)):
        if period == 0:
            turnover = 0.0
        elif _should_rebalance(period, current, target_array, spec):
            # One-way turnover: half the sum of absolute weight changes, since
            # every sale funds a purchase.
            turnover = float(np.abs(target_array - current).sum() / 2.0)
            current = target_array.copy()
        else:
            turnover = 0.0

        cost = turnover * spec.cost_bps / 10_000.0

        period_returns = rets.iloc[period].to_numpy()
        gross = float(current @ period_returns)

        out_returns.append(gross - cost)
        weight_rows.append(current.copy())
        turnovers.append(turnover)
        costs.append(cost)

        # Drift: each holding grows by its own return, then renormalise.
        grown = current * (1.0 + period_returns)
        total = grown.sum()
        current = grown / total if total > 0 else current

    index = rets.index
    return RebalancePath(
        returns=pd.Series(out_returns, index=index),
        weights=pd.DataFrame(weight_rows, index=index, columns=panel.assets),
        turnover=pd.Series(turnovers, index=index),
        costs=pd.Series(costs, index=index),
        targets=target,
        spec=spec,
    )


def simulate_with_sleeve(
    panel: ReturnPanel,
    weights: pd.Series | dict[str, float],
    gold_weight: float,
    spec: RebalanceSpec | Rebalance | None = None,
) -> RebalancePath:
    """Hold an allocation whose commodities bucket is itself a two-part sleeve.

    The sleeve is rebalanced on the same schedule as the portfolio, as a
    separate operation: the portfolio trades back to target across its buckets,
    and inside the commodities bucket gold and ex-gold trade back to the
    slider's ratio.

    Using one setting for both is deliberate. If the sleeve rebalanced on a
    different schedule -- or always, or never -- the slider would stop meaning
    what it says the moment the user changed an unrelated toggle. With
    rebalancing off, the sleeve drifts too, and a strong run in gold leaves the
    sleeve gold-heavier than the slider states. That is the honest consequence
    of choosing not to rebalance, and it should be visible rather than quietly
    corrected.

    Implemented by expanding the sleeve weight into its two components and
    simulating on the underlying panel, so the sleeve's drift and the
    portfolio's drift are modelled by the same mechanism rather than two.
    """
    from core.sleeve import COMMODITIES_EX_GOLD, GOLD, SLEEVE

    if not 0.0 <= gold_weight <= 1.0:
        raise ValueError(f"gold_weight must be between 0 and 1, got {gold_weight}")

    supplied = pd.Series(weights, dtype=float)
    if SLEEVE not in supplied.index:
        raise KeyError(
            f"No weight supplied for {SLEEVE!r}; use `simulate` for panels "
            f"without a commodities sleeve"
        )
    missing = [c for c in (GOLD, COMMODITIES_EX_GOLD) if c not in panel.assets]
    if missing:
        raise KeyError(f"Panel is missing sleeve components {missing}")

    sleeve_weight = float(supplied[SLEEVE])
    expanded = supplied.drop(index=[SLEEVE]).to_dict()
    expanded[GOLD] = sleeve_weight * gold_weight
    expanded[COMMODITIES_EX_GOLD] = sleeve_weight * (1.0 - gold_weight)

    return simulate(panel, expanded, spec)


def compare_schedules(
    panel: ReturnPanel,
    weights: pd.Series | dict[str, float],
    cost_bps: float = DEFAULT_COST_BPS,
    threshold_band: float = DEFAULT_THRESHOLD_BAND,
    risk_free: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Run one allocation under every schedule and tabulate the outcomes.

    The comparison a user needs before choosing a setting: what each one
    earned, what it risked, how much it traded and what that trading cost.
    """
    from core.portfolio import portfolio_stats

    if risk_free is None and "cash" in panel.assets:
        risk_free = panel.returns["cash"]

    rows = []
    for schedule in Rebalance:
        spec = RebalanceSpec(
            schedule=schedule, cost_bps=cost_bps, threshold_band=threshold_band
        )
        path = simulate(panel, weights, spec)
        stats = measure_path(panel, path, risk_free=risk_free)

        rows.append(
            {
                "schedule": schedule.value,
                "return": stats.realised_return,
                "vol": stats.volatility,
                "sharpe": stats.sharpe,
                "sortino": stats.sortino,
                "max_dd": stats.max_drawdown,
                "n_trades": path.n_rebalances,
                "total_turnover": path.total_turnover,
                "total_cost": path.total_cost,
                "max_drift": path.max_drift,
            }
        )

    return pd.DataFrame(rows).set_index("schedule")


def measure_path(
    panel: ReturnPanel,
    path: RebalancePath,
    risk_free: pd.Series | float | None = None,
):
    """Measure a simulated path with the Step 2 statistics.

    Builds a one-asset panel from the path's realised returns so the existing
    measurement code applies unchanged. Every schedule is therefore measured by
    identical logic -- no setting gets its own statistics implementation, which
    is how they would eventually diverge.
    """
    from core.portfolio import portfolio_stats

    if risk_free is None and "cash" in panel.assets:
        risk_free = panel.returns["cash"]

    wrapper = ReturnPanel(
        pd.DataFrame({"portfolio": path.returns}), panel.periods_per_year
    )
    return portfolio_stats(wrapper, {"portfolio": 1.0}, risk_free=risk_free)
