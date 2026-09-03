"""Mandates: a return target, a risk budget, and policy limits.

The optimizers in `optimize.py` answer "what was best" on one measure. That is
not how an allocation is actually commissioned. A real mandate reads closer to:

    Achieve at least 6% a year, with volatility no more than 10%, holding at
    least 5% cash, no more than 20% private equity, and no more than 60% in
    growth assets.

Three parts -- a target, a budget, and constraints -- and the question is not
"which allocation scores highest" but "which allocations satisfy all of this,
and is there even one".

Two things follow, and both are why this module exists rather than another
objective in `optimize.py`.

**Volatility is a limit, not a trade-off.** A Sharpe optimizer will happily
accept 12% volatility to earn more, because the ratio improved. A mandate
saying "no more than 10%" means exactly that, and no amount of extra return
makes 12% acceptable.

**Infeasibility is a real answer.** Over some periods the target simply cannot
be reached inside the budget with these assets, and saying so is useful. But
"impossible" on its own is not actionable, so when a mandate fails this module
reports what would have to change: how far the target would have to fall, how
far the budget would have to rise, or which constraint is doing the blocking
and how far it would need to relax.

When a mandate succeeds there is usually not one answer but hundreds. This
module returns all of them and leaves the ranking to whoever holds the
mandate -- lowest drawdown, highest return, most headroom under the budget are
all defensible, and which one matters is not a question the code can settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from core.constraints import Constraints, GroupLimit, project_onto_constraints
from core.optimize import DEFAULT_SAMPLES, _measure_samples, sample_allocations
from core.panels import ReturnPanel
from core.config import Rebalance
from core.rebalance import RebalanceSpec, simulate

# Columns a qualifying allocation can be ranked by, and which direction is
# better. The user chooses; the code only records what "better" means.
# Monthly rebalancing with no trading cost is what makes the weighted sum of
# asset returns the portfolio return, which is the assumption every other
# module here already makes. It is also the only setting the vectorised
# measurement can use: any other schedule has to simulate each allocation
# individually, which is orders of magnitude slower. So it is the default, and
# choosing otherwise is a deliberate and visibly costlier decision.
MONTHLY_NO_COST = RebalanceSpec(schedule=Rebalance.MONTHLY, cost_bps=0.0)

RANKABLE: dict[str, bool] = {
    "realised_return": True,
    "volatility": False,
    "sharpe": True,
    "sortino": True,
    "max_drawdown": True,  # closer to zero is better, and drawdowns are negative
    "months_to_recover": False,
    "months_underwater": False,
}


@dataclass(frozen=True)
class Mandate:
    """What an allocation is required to achieve, and within what limits.

    `rebalance` records how the allocation is to be held. It is part of the
    mandate rather than a display setting: "at least 6% a year" means something
    different for a portfolio corrected annually than for one left to drift,
    and an instruction that does not say which is incomplete.
    """

    target_return: float | None = None
    max_volatility: float | None = None
    max_drawdown: float | None = None
    max_recovery_months: int | None = None
    constraints: Constraints = field(default_factory=Constraints)
    rebalance: RebalanceSpec = field(default_factory=lambda: MONTHLY_NO_COST)

    def __post_init__(self) -> None:
        if self.target_return is not None and not -1.0 < self.target_return < 5.0:
            raise ValueError(
                f"target_return should be a fraction, e.g. 0.06 for 6%; "
                f"got {self.target_return}"
            )
        if self.max_volatility is not None and not 0.0 < self.max_volatility < 5.0:
            raise ValueError(
                f"max_volatility should be a fraction, e.g. 0.10 for 10%; "
                f"got {self.max_volatility}"
            )
        if self.max_drawdown is not None and not -1.0 <= self.max_drawdown <= 0.0:
            raise ValueError(
                f"max_drawdown should be negative, e.g. -0.20 for a 20% limit; "
                f"got {self.max_drawdown}"
            )
        if self.max_recovery_months is not None and self.max_recovery_months < 1:
            raise ValueError(
                f"max_recovery_months must be at least 1, got "
                f"{self.max_recovery_months}"
            )
        if (
            self.target_return is None
            and self.max_volatility is None
            and self.max_drawdown is None
            and self.max_recovery_months is None
        ):
            raise ValueError(
                "A mandate needs at least one requirement. With none, every "
                "allocation qualifies and there is nothing to solve."
            )

    def describe(self) -> str:
        parts = []
        if self.target_return is not None:
            parts.append(f"return >= {self.target_return:.1%}")
        if self.max_volatility is not None:
            parts.append(f"volatility <= {self.max_volatility:.1%}")
        if self.max_drawdown is not None:
            parts.append(f"drawdown no worse than {self.max_drawdown:.1%}")
        if self.max_recovery_months is not None:
            parts.append(f"recovery within {self.max_recovery_months} months")
        if not self.constraints.is_empty:
            parts.append(self.constraints.describe())
        parts.append(self.rebalance.describe())
        return "; ".join(parts)

    def qualifies(self, measured: pd.DataFrame, ignoring: str | None = None) -> pd.Series:
        """Boolean mask over a measured sample set.

        `ignoring` drops one requirement from the test, which is how the
        diagnosis asks "what would be reachable if this limit were lifted".
        Done here rather than by building a mandate with that field set to
        None, because a mandate with every requirement removed is invalid --
        and a mandate with exactly one requirement would hit that case.
        """
        mask = pd.Series(True, index=measured.index)
        if self.target_return is not None and ignoring != "target_return":
            mask &= measured["realised_return"] >= self.target_return
        if self.max_volatility is not None and ignoring != "max_volatility":
            mask &= measured["volatility"] <= self.max_volatility
        if self.max_drawdown is not None and ignoring != "max_drawdown":
            mask &= measured["max_drawdown"] >= self.max_drawdown
        if (
            self.max_recovery_months is not None
            and ignoring != "max_recovery_months"
        ):
            mask &= measured["months_to_recover"] <= self.max_recovery_months
        return mask


@dataclass(frozen=True)
class Relaxation:
    """One change that would make an infeasible mandate achievable."""

    what: str
    current: float
    required: float
    note: str = ""

    def describe(self) -> str:
        tail = f" ({self.note})" if self.note else ""
        return (
            f"{self.what}: {self.current:.1%} -> {self.required:.1%}{tail}"
        )


@dataclass(frozen=True)
class MandateResult:
    """Everything that qualified, or why nothing did."""

    mandate: Mandate
    qualifying: pd.DataFrame
    assets: list[str]
    n_sampled: int
    relaxations: tuple[Relaxation, ...] = ()

    @property
    def feasible(self) -> bool:
        return len(self.qualifying) > 0

    @property
    def n_qualifying(self) -> int:
        return len(self.qualifying)

    def ranked(
        self,
        by: str = "max_drawdown",
        limit: int | None = None,
        resolution: float | None = None,
    ) -> pd.DataFrame:
        """The qualifying allocations, sorted however the holder prefers.

        Several hundred allocations routinely satisfy a mandate. Which is
        "best" among them depends on what the holder cares about, and that is
        not a question the code can settle -- so it ranks on request rather
        than choosing.

        `resolution` collapses allocations that are the same portfolio to
        anyone deciding. The qualifying set is near-continuous, so the top
        twelve by any measure are usually neighbours differing in the third
        decimal: 40.2% equity and 40.3% equity are not two options, and
        presenting them as such offers a choice that does not exist.

        Rounding is used for grouping only. The figures reported are the exact
        ones from the best member of each group, so nothing is distorted -- the
        rounding decides which rows are *the same*, not what they are.
        """
        if not self.feasible:
            return self.qualifying

        if by not in RANKABLE:
            raise KeyError(
                f"Cannot rank by {by!r}. Available: {sorted(RANKABLE)}"
            )

        ordered = self.qualifying.sort_values(by, ascending=not RANKABLE[by])

        if resolution:
            if not 0.0 < resolution <= 0.5:
                raise ValueError(
                    f"resolution must be a fraction between 0 and 0.5 "
                    f"(0.05 for five percentage points), got {resolution}"
                )
            # Already sorted, so the first row of each group is its best
            # member. Keeping that one means the figures shown belong to a
            # real allocation rather than to a rounded average of several.
            buckets = (ordered[self.assets] / resolution).round().astype(int)
            ordered = ordered[~buckets.duplicated()]

        return ordered.head(limit) if limit else ordered

    def distinct_count(self, resolution: float) -> int:
        """How many meaningfully different allocations qualify.

        Usually far fewer than the raw count. A mandate met by 3,800 sampled
        allocations may be met by a dozen genuinely different portfolios, and
        the second number is the one worth reporting.
        """
        if not self.feasible:
            return 0
        buckets = (self.qualifying[self.assets] / resolution).round().astype(int)
        return int((~buckets.duplicated()).sum())

    def envelope(self) -> pd.DataFrame:
        """The range each asset's weight takes across qualifying allocations.

        The honest shape of the answer: not one allocation, but the space of
        allocations that meet the mandate. An asset that ranges from 0% to 60%
        is one the mandate has no opinion about.
        """
        if not self.feasible:
            return pd.DataFrame()

        weights = self.qualifying[self.assets]
        return pd.DataFrame(
            {
                "min": weights.min(),
                "median": weights.median(),
                "max": weights.max(),
                "spread": weights.max() - weights.min(),
            }
        )

    def headroom(self) -> pd.DataFrame:
        """How much slack each qualifying allocation has against each limit.

        An allocation that only just clears the volatility budget is a
        different proposition from one comfortably inside it, even though both
        satisfy the mandate.
        """
        if not self.feasible:
            return pd.DataFrame()

        out = pd.DataFrame(index=self.qualifying.index)
        if self.mandate.target_return is not None:
            out["return_headroom"] = (
                self.qualifying["realised_return"] - self.mandate.target_return
            )
        if self.mandate.max_volatility is not None:
            out["volatility_headroom"] = (
                self.mandate.max_volatility - self.qualifying["volatility"]
            )
        if self.mandate.max_drawdown is not None:
            out["drawdown_headroom"] = (
                self.qualifying["max_drawdown"] - self.mandate.max_drawdown
            )
        return out

    def explain(self) -> str:
        if self.feasible:
            return (
                f"{self.n_qualifying:,} of {self.n_sampled:,} allocations meet "
                f"the mandate. Rank them by whichever measure matters."
            )
        if not self.relaxations:
            return "No allocation meets the mandate, and no single change fixes it."
        lines = ["No allocation meets the mandate. Any one of these would help:"]
        lines.extend(f"  - {r.describe()}" for r in self.relaxations)
        return "\n".join(lines)


def _measure_under_rebalancing(
    panel: ReturnPanel,
    samples: np.ndarray,
    spec: RebalanceSpec,
    risk_free: pd.Series | float | None,
) -> pd.DataFrame:
    """Measure each sampled allocation as it would actually have been held.

    The vectorised sampler assumes monthly rebalancing, which is what makes the
    weighted sum of asset returns the portfolio return. Any other schedule
    produces a different path, so each allocation has to be simulated.

    That is far slower -- a loop rather than one matrix product -- so it runs
    only when the mandate asks for something other than monthly. A mandate that
    does not say how the portfolio is held is incomplete, but the common case
    should not pay for the general one.
    """
    from core.optimize import _COLUMN_FOR  # noqa: F401  (kept for symmetry)
    from core.portfolio import portfolio_stats

    assets = panel.assets
    rows = []

    for row in samples:
        weights = pd.Series(row, index=assets)
        path = simulate(panel, weights, spec)
        held = ReturnPanel(
            pd.DataFrame({"portfolio": path.returns}), panel.periods_per_year
        )
        stats = portfolio_stats(held, {"portfolio": 1.0}, risk_free=risk_free)

        record = dict(zip(assets, row))
        record.update(
            {
                "realised_return": stats.realised_return,
                "volatility": stats.volatility,
                "sharpe": stats.sharpe,
                "sortino": stats.sortino,
                "max_drawdown": stats.max_drawdown,
                "months_underwater": stats.drawdown.months_underwater,
                "months_to_recover": (
                    float(stats.drawdown.months_to_recover)
                    if stats.drawdown.recovered
                    else np.inf
                ),
                "turnover": path.total_turnover,
                "trading_cost": path.total_cost,
            }
        )
        rows.append(record)

    return pd.DataFrame(rows)


def _measure(
    panel: ReturnPanel,
    samples: np.ndarray,
    spec: RebalanceSpec,
    risk_free: pd.Series | float | None,
) -> pd.DataFrame:
    """Measure samples, using the fast path where the schedule allows it."""
    if spec.schedule is Rebalance.MONTHLY and spec.cost_bps == 0.0:
        return _measure_samples(panel, samples, risk_free)
    return _measure_under_rebalancing(panel, samples, spec, risk_free)


def _feasible_samples(
    panel: ReturnPanel,
    constraints: Constraints,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    raw = sample_allocations(len(panel.assets), n_samples, seed=seed)
    if constraints.is_empty:
        return raw
    return project_onto_constraints(raw, panel.assets, constraints)


def solve_mandate(
    panel: ReturnPanel,
    mandate: Mandate,
    risk_free: pd.Series | float | None = None,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
    diagnose: bool = True,
) -> MandateResult:
    """Find every allocation satisfying the mandate, or explain why none does.

    `diagnose=False` skips the relaxation analysis, which costs nothing extra
    when the mandate is already feasible but does re-solve several times when
    it is not.
    """
    mandate.constraints.validate(panel.assets)

    if risk_free is None and "cash" in panel.assets:
        risk_free = panel.returns["cash"]

    samples = _feasible_samples(panel, mandate.constraints, n_samples, seed)
    if len(samples) == 0:
        raise RuntimeError(
            f"No allocation satisfies the policy limits themselves: "
            f"{mandate.constraints.describe()}. The limits may be jointly "
            f"impossible even though each is individually valid."
        )

    measured = _measure(panel, samples, mandate.rebalance, risk_free)
    qualifying = measured[mandate.qualifies(measured)].copy()

    relaxations: tuple[Relaxation, ...] = ()
    if len(qualifying) == 0 and diagnose:
        relaxations = diagnose_infeasibility(
            panel, mandate, measured, risk_free, n_samples, seed
        )

    return MandateResult(
        mandate=mandate,
        qualifying=qualifying,
        assets=panel.assets,
        n_sampled=len(measured),
        relaxations=relaxations,
    )


def diagnose_infeasibility(
    panel: ReturnPanel,
    mandate: Mandate,
    measured: pd.DataFrame,
    risk_free: pd.Series | float | None,
    n_samples: int,
    seed: int,
) -> tuple[Relaxation, ...]:
    """Work out what would have to change for the mandate to be achievable.

    Each requirement is relaxed in turn, holding the others, and the minimum
    change that admits at least one allocation is reported. "Impossible" is a
    true answer but not an actionable one; "your target would have to drop to
    4.8%, or your volatility budget rise to 12.3%, or your private equity cap
    relax to 35%" lets the holder decide which rule to argue with.

    Constraints are tested by removing each in turn, since the useful question
    is which rule is blocking rather than the exact number it would need.
    """
    relaxations: list[Relaxation] = []

    # What return is reachable if the target alone is dropped?
    if mandate.target_return is not None:
        others = measured[mandate.qualifies(measured, ignoring="target_return")]
        if len(others) > 0:
            best = float(others["realised_return"].max())
            relaxations.append(
                Relaxation(
                    "target return",
                    mandate.target_return,
                    best,
                    "the most the other limits allow",
                )
            )

    # What volatility budget would admit the target?
    if mandate.max_volatility is not None:
        others = measured[mandate.qualifies(measured, ignoring="max_volatility")]
        if len(others) > 0:
            needed = float(others["volatility"].min())
            relaxations.append(
                Relaxation(
                    "volatility budget",
                    mandate.max_volatility,
                    needed,
                    "the calmest allocation that hits the target",
                )
            )

    # What drawdown limit would admit the target?
    if mandate.max_drawdown is not None:
        others = measured[mandate.qualifies(measured, ignoring="max_drawdown")]
        if len(others) > 0:
            needed = float(others["max_drawdown"].max())
            relaxations.append(
                Relaxation(
                    "drawdown limit",
                    mandate.max_drawdown,
                    needed,
                    "the shallowest that meets the other limits",
                )
            )

    # Which policy constraint is blocking?
    constraints = mandate.constraints
    if not constraints.is_empty:
        for label, relaxed in _constraint_variants(constraints):
            trial = replace(mandate, constraints=relaxed)
            samples = _feasible_samples(panel, relaxed, n_samples, seed)
            if len(samples) == 0:
                continue
            trial_measured = _measure(panel, samples, mandate.rebalance, risk_free)
            if trial.qualifies(trial_measured).any():
                relaxations.append(
                    Relaxation(
                        f"drop the {label} rule",
                        0.0,
                        0.0,
                        "removing this alone makes the mandate achievable",
                    )
                )

    if not relaxations:
        # Nothing single-handedly fixes it. Report the closest miss so the
        # holder can see how far away they are rather than only that they are.
        closest = _closest_miss(mandate, measured)
        if closest is not None:
            relaxations.append(closest)

    return tuple(relaxations)


def _constraint_variants(
    constraints: Constraints,
) -> list[tuple[str, Constraints]]:
    """Each constraint removed in turn, labelled."""
    variants = []
    for asset, cap in constraints.caps.items():
        variants.append(
            (
                f"{asset} <= {cap:.0%}",
                Constraints(
                    caps={k: v for k, v in constraints.caps.items() if k != asset},
                    floors=constraints.floors,
                    groups=constraints.groups,
                ),
            )
        )
    for asset, floor in constraints.floors.items():
        variants.append(
            (
                f"{asset} >= {floor:.0%}",
                Constraints(
                    caps=constraints.caps,
                    floors={k: v for k, v in constraints.floors.items() if k != asset},
                    groups=constraints.groups,
                ),
            )
        )
    for group in constraints.groups:
        variants.append(
            (
                group.name,
                Constraints(
                    caps=constraints.caps,
                    floors=constraints.floors,
                    groups=tuple(g for g in constraints.groups if g is not group),
                ),
            )
        )
    return variants


def _closest_miss(mandate: Mandate, measured: pd.DataFrame) -> Relaxation | None:
    """How near the best available allocation came, when nothing qualified.

    Reported when no single relaxation is enough, because the requirements
    conflict jointly rather than individually. Knowing the gap is small is very
    different from knowing it is large.
    """
    if measured.empty:
        return None

    if mandate.target_return is not None:
        best = float(measured["realised_return"].max())
        return Relaxation(
            "target return",
            mandate.target_return,
            best,
            "several limits conflict; this is the most reachable at all",
        )
    if mandate.max_volatility is not None:
        return Relaxation(
            "volatility budget",
            mandate.max_volatility,
            float(measured["volatility"].min()),
            "several limits conflict; this is the calmest available",
        )
    return None


def frontier_of_mandates(
    panel: ReturnPanel,
    targets: list[float],
    max_volatility: float,
    constraints: Constraints | None = None,
    risk_free: pd.Series | float | None = None,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """Which return targets are reachable within one risk budget.

    Sweeping the target shows where the mandate stops being achievable, which
    is more useful than testing a single number: it answers "how much could I
    have asked for" rather than only "could I have asked for this".
    """
    constraints = constraints or Constraints()

    rows = []
    for target in targets:
        mandate = Mandate(
            target_return=target,
            max_volatility=max_volatility,
            constraints=constraints,
        )
        result = solve_mandate(
            panel, mandate, risk_free, n_samples, seed, diagnose=False
        )
        row = {
            "target": target,
            "feasible": result.feasible,
            "n_qualifying": result.n_qualifying,
        }
        if result.feasible:
            best = result.ranked("max_drawdown").iloc[0]
            row.update(
                {
                    "best_drawdown": best["max_drawdown"],
                    "return_at_best": best["realised_return"],
                    "vol_at_best": best["volatility"],
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Across periods
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobustMandateResult:
    """Which allocations met the mandate in every period, not merely in one."""

    mandate: Mandate
    per_period: dict[str, pd.DataFrame]
    survivors: pd.DataFrame
    assets: list[str]
    period_labels: list[str]

    @property
    def any_survivors(self) -> bool:
        return len(self.survivors) > 0

    def survival_counts(self) -> pd.Series:
        """How many allocations met the mandate in each period on its own.

        A period where almost nothing qualified is the one doing the work. It
        is usually a crisis, and it is where a mandate is actually tested.
        """
        return pd.Series(
            {label: len(frame) for label, frame in self.per_period.items()}
        )

    def explain(self) -> str:
        counts = self.survival_counts()
        hardest = counts.idxmin()

        if self.any_survivors:
            return (
                f"{len(self.survivors):,} allocations met the mandate in all "
                f"{len(self.period_labels)} periods. The binding period was "
                f"{hardest} ({counts[hardest]:,} qualified there)."
            )
        return (
            f"No allocation met the mandate in every period. The binding "
            f"period was {hardest}, where {counts[hardest]:,} of the sampled "
            f"allocations qualified. Meeting a mandate once is a hindsight "
            f"result; meeting it across regimes is the harder and more "
            f"meaningful claim."
        )


def solve_mandate_across_periods(
    panel: ReturnPanel,
    mandate: Mandate,
    periods: list,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
    tolerance: float = 1e-9,
) -> RobustMandateResult:
    """Find allocations that met the mandate in *every* period.

    Meeting a mandate over one window is a hindsight result: the allocation was
    chosen knowing what happened. Meeting it in the crisis and the recovery and
    the inflation shock is a substantially stronger claim, because no single
    regime could have been anticipated.

    The same sampled allocations are evaluated in each period, so an allocation
    can be tracked across them. Sampling independently per period would produce
    sets that cannot be intersected.

    Expect few survivors, or none. That is the point rather than a failure --
    an empty result says the mandate was not achievable through the whole
    sample, which is worth knowing before committing to it.
    """
    mandate.constraints.validate(panel.assets)

    samples = _feasible_samples(panel, mandate.constraints, n_samples, seed)
    if len(samples) == 0:
        raise RuntimeError(
            f"No allocation satisfies the policy limits: "
            f"{mandate.constraints.describe()}"
        )

    per_period: dict[str, pd.DataFrame] = {}
    surviving = np.ones(len(samples), dtype=bool)

    for period in periods:
        window = panel.between(period.start, period.end)
        cash = window.returns["cash"] if "cash" in window.assets else None

        measured = _measure(window, samples, mandate.rebalance, cash)
        mask = mandate.qualifies(measured).to_numpy()

        per_period[period.label] = measured[mask].copy()
        surviving &= mask

    survivors = pd.DataFrame(samples[surviving], columns=panel.assets)

    return RobustMandateResult(
        mandate=mandate,
        per_period=per_period,
        survivors=survivors,
        assets=panel.assets,
        period_labels=[p.label for p in periods],
    )
