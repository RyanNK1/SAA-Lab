"""Policy constraints, and what they cost.

No real allocation is unconstrained. There is always a minimum cash holding, a
cap on alternatives, a floor on fixed income -- sometimes for good reasons,
sometimes because a committee wrote it down years ago and nobody revisited it.

Every one of those rules costs return. Almost nobody measures how much.

This module answers that: solve the problem without the constraints, solve it
with them, and report the gap.

    Your 20% cap on private equity cost 34bps a year over this period.

That sentence is what strategic asset allocation work actually delivers, and
it reframes the tool from "here is the answer" to "here is what your rules are
costing you" -- which is the question a committee genuinely has.

The cost is always non-negative. A constraint shrinks the feasible set, and a
smaller set cannot contain a better answer, so a negative cost would mean the
optimizer failed rather than that the constraint helped. That identity is a
free correctness check and is enforced.

Constraints also fix a degenerate result from the optimizer. Asked to minimise
drawdown with no restrictions, it puts everything in cash -- a correct answer
to the question asked and a useless answer to the question meant. A floor on
risky assets forces it to choose something a person might actually hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import Objective
from core.optimize import (
    DEFAULT_SAMPLES,
    is_better,
    DEFAULT_TOLERANCE,
    Method,
    OptimizationResult,
    _measure_samples,
    _near_optimal,
    _refine,
    _COLUMN_FOR,
    _MAXIMISE,
    sample_allocations,
    score,
)
from core.panels import ReturnPanel
from core.portfolio import portfolio_stats

_TOL = 1e-9


@dataclass(frozen=True)
class GroupLimit:
    """A ceiling or floor on several assets taken together.

    Real mandates are often written this way -- "growth assets no more than
    60%" -- rather than asset by asset, because the concern is total exposure
    to a kind of risk, not to a particular label.
    """

    name: str
    assets: tuple[str, ...]
    maximum: float | None = None
    minimum: float | None = None

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError(f"Group {self.name!r} has no assets")
        if self.maximum is None and self.minimum is None:
            raise ValueError(f"Group {self.name!r} sets neither a floor nor a cap")
        for bound, label in ((self.maximum, "maximum"), (self.minimum, "minimum")):
            if bound is not None and not 0.0 <= bound <= 1.0:
                raise ValueError(
                    f"Group {self.name!r} {label} must be between 0 and 1, "
                    f"got {bound}"
                )
        if (
            self.maximum is not None
            and self.minimum is not None
            and self.minimum > self.maximum + _TOL
        ):
            raise ValueError(
                f"Group {self.name!r} floor {self.minimum} exceeds its cap "
                f"{self.maximum}"
            )


@dataclass(frozen=True)
class Constraints:
    """The policy limits applied to an allocation.

    `caps` and `floors` are per asset. `groups` are joint limits. Anything not
    named is unconstrained beyond long-only and fully invested.
    """

    caps: dict[str, float] = field(default_factory=dict)
    floors: dict[str, float] = field(default_factory=dict)
    groups: tuple[GroupLimit, ...] = ()

    def __post_init__(self) -> None:
        for name, value in {**self.caps, **self.floors}.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Limit for {name!r} must be between 0 and 1, got {value}"
                )
        for asset, floor in self.floors.items():
            cap = self.caps.get(asset)
            if cap is not None and floor > cap + _TOL:
                raise ValueError(
                    f"Floor {floor:.1%} for {asset!r} exceeds its cap {cap:.1%}"
                )

    @property
    def is_empty(self) -> bool:
        return not self.caps and not self.floors and not self.groups

    def validate(self, assets: list[str]) -> None:
        """Check the constraints can be satisfied at all, before optimising.

        An infeasible set produces a solver failure deep in the stack, or
        worse, a plausible-looking answer that quietly violates a rule. Caught
        here, the message names the problem.
        """
        unknown = (set(self.caps) | set(self.floors)) - set(assets)
        for group in self.groups:
            unknown |= set(group.assets) - set(assets)
        if unknown:
            raise KeyError(
                f"Constraints reference assets not in the panel: {sorted(unknown)}. "
                f"Panel has: {assets}"
            )

        total_floor = sum(self.floors.values())
        if total_floor > 1.0 + _TOL:
            raise ValueError(
                f"Floors sum to {total_floor:.1%}, which is more than the whole "
                f"portfolio. No allocation can satisfy them."
            )

        total_cap = sum(self.caps.get(a, 1.0) for a in assets)
        if total_cap < 1.0 - _TOL:
            raise ValueError(
                f"Caps sum to {total_cap:.1%}, so the portfolio cannot be fully "
                f"invested. Raise a cap or add an uncapped asset."
            )

        for group in self.groups:
            if group.minimum is not None:
                reachable = sum(self.caps.get(a, 1.0) for a in group.assets)
                if reachable < group.minimum - _TOL:
                    raise ValueError(
                        f"Group {group.name!r} needs at least "
                        f"{group.minimum:.1%} but its assets are capped at "
                        f"{reachable:.1%} combined"
                    )
            if group.maximum is not None:
                forced = sum(self.floors.get(a, 0.0) for a in group.assets)
                if forced > group.maximum + _TOL:
                    raise ValueError(
                        f"Group {group.name!r} is capped at {group.maximum:.1%} "
                        f"but its assets have floors totalling {forced:.1%}"
                    )

        outside_floors = sum(
            self.floors.get(a, 0.0) for a in assets
        )
        for group in self.groups:
            if group.maximum is None:
                continue
            others = [a for a in assets if a not in group.assets]
            headroom = sum(self.caps.get(a, 1.0) for a in others) + group.maximum
            if headroom < 1.0 - _TOL:
                raise ValueError(
                    f"Group {group.name!r} capped at {group.maximum:.1%} leaves "
                    f"only {headroom:.1%} investable in total"
                )
        del outside_floors

    def satisfied_by(self, weights: pd.Series, tol: float = 1e-6) -> bool:
        for asset, cap in self.caps.items():
            if weights.get(asset, 0.0) > cap + tol:
                return False
        for asset, floor in self.floors.items():
            if weights.get(asset, 0.0) < floor - tol:
                return False
        for group in self.groups:
            total = float(sum(weights.get(a, 0.0) for a in group.assets))
            if group.maximum is not None and total > group.maximum + tol:
                return False
            if group.minimum is not None and total < group.minimum - tol:
                return False
        return True

    def violations(self, weights: pd.Series, tol: float = 1e-6) -> list[str]:
        """Human-readable list of what a given allocation breaks."""
        problems = []
        for asset, cap in self.caps.items():
            held = weights.get(asset, 0.0)
            if held > cap + tol:
                problems.append(f"{asset} at {held:.1%} exceeds its {cap:.1%} cap")
        for asset, floor in self.floors.items():
            held = weights.get(asset, 0.0)
            if held < floor - tol:
                problems.append(f"{asset} at {held:.1%} is below its {floor:.1%} floor")
        for group in self.groups:
            total = float(sum(weights.get(a, 0.0) for a in group.assets))
            if group.maximum is not None and total > group.maximum + tol:
                problems.append(
                    f"{group.name} at {total:.1%} exceeds its {group.maximum:.1%} cap"
                )
            if group.minimum is not None and total < group.minimum - tol:
                problems.append(
                    f"{group.name} at {total:.1%} is below its "
                    f"{group.minimum:.1%} floor"
                )
        return problems

    def describe(self) -> str:
        if self.is_empty:
            return "unconstrained"
        parts = []
        for asset, cap in sorted(self.caps.items()):
            parts.append(f"{asset} <= {cap:.0%}")
        for asset, floor in sorted(self.floors.items()):
            parts.append(f"{asset} >= {floor:.0%}")
        for group in self.groups:
            if group.maximum is not None:
                parts.append(f"{group.name} <= {group.maximum:.0%}")
            if group.minimum is not None:
                parts.append(f"{group.name} >= {group.minimum:.0%}")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Feasible sampling
# ---------------------------------------------------------------------------

def project_onto_constraints(
    samples: np.ndarray, assets: list[str], constraints: Constraints
) -> np.ndarray:
    """Push allocations into the feasible region, keeping them summing to one.

    Floors are applied first and held: the remaining budget is distributed
    across the headroom above each floor, respecting caps. Group limits are
    then enforced by scaling the group and redistributing outside it. Repeated
    a few times, since satisfying one limit can breach another.

    Rows that remain infeasible are dropped rather than returned slightly
    wrong -- a sample that violates a rule would let the search report an
    allocation the user explicitly forbade.
    """
    index = {a: i for i, a in enumerate(assets)}
    n = len(assets)

    floors = np.zeros(n)
    caps = np.ones(n)
    for asset, floor in constraints.floors.items():
        floors[index[asset]] = floor
    for asset, cap in constraints.caps.items():
        caps[index[asset]] = cap

    w = np.clip(samples.copy(), floors, caps)

    for _ in range(24):
        # Restore the budget, moving only within the room each asset has.
        deficit = 1.0 - w.sum(axis=1)

        room_up = caps - w
        room_down = w - floors
        room = np.where(deficit[:, None] > 0, room_up, room_down)
        available = room.sum(axis=1)

        share = np.divide(
            room, available[:, None], out=np.zeros_like(room), where=available[:, None] > 1e-15
        )
        w = np.clip(w + share * deficit[:, None], floors, caps)

        # Group limits.
        for group in constraints.groups:
            columns = [index[a] for a in group.assets]
            others = [i for i in range(n) if i not in columns]
            if not others:
                continue

            total = w[:, columns].sum(axis=1)

            for bound, breached in (
                (group.maximum, None if group.maximum is None else total > group.maximum + _TOL),
                (group.minimum, None if group.minimum is None else total < group.minimum - _TOL),
            ):
                if bound is None or not breached.any():
                    continue
                rows = np.where(breached)[0]
                current = total[rows]
                scale = np.divide(
                    bound, current, out=np.ones_like(current), where=current > 1e-15
                )
                w[np.ix_(rows, columns)] = np.clip(
                    w[np.ix_(rows, columns)] * scale[:, None],
                    floors[columns],
                    caps[columns],
                )
                shortfall = 1.0 - w[rows].sum(axis=1)
                other_room = (caps[others] - w[np.ix_(rows, others)])
                other_available = other_room.sum(axis=1)
                other_share = np.divide(
                    other_room,
                    other_available[:, None],
                    out=np.zeros_like(other_room),
                    where=other_available[:, None] > 1e-15,
                )
                w[np.ix_(rows, others)] = np.clip(
                    w[np.ix_(rows, others)] + other_share * shortfall[:, None],
                    floors[others],
                    caps[others],
                )

        if np.abs(1.0 - w.sum(axis=1)).max() < 1e-10:
            feasible = np.array(
                [
                    constraints.satisfied_by(pd.Series(row, index=assets))
                    for row in w
                ]
            )
            if feasible.all():
                break

    keep = np.abs(1.0 - w.sum(axis=1)) < 1e-8
    keep &= np.array(
        [constraints.satisfied_by(pd.Series(row, index=assets)) for row in w]
    )
    return w[keep]


# ---------------------------------------------------------------------------
# Constrained optimisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintCost:
    """What a set of constraints cost, against the unconstrained optimum."""

    objective: Objective
    unconstrained: OptimizationResult
    constrained: OptimizationResult
    constraints: Constraints

    @property
    def cost(self) -> float:
        """How much worse the constrained answer is, in the objective's units.

        Always non-negative: constraints shrink the feasible set, and a smaller
        set cannot contain a better answer. A negative value means the
        optimizer failed, not that the constraint helped.
        """
        free = self.unconstrained.value
        bound = self.constrained.value

        if _MAXIMISE[self.objective] or self.objective is Objective.MIN_DRAWDOWN:
            return free - bound
        return bound - free

    @property
    def return_cost(self) -> float:
        """The difference in annualised return. The number a committee wants."""
        return (
            self.unconstrained.stats.realised_return
            - self.constrained.stats.realised_return
        )

    def describe(self) -> str:
        if self.constraints.is_empty:
            return "no constraints applied"
        return (
            f"{self.constraints.describe()} cost "
            f"{abs(self.return_cost) * 10_000:.0f}bps a year "
            f"({self.objective.value} fell by {self.cost:.4f})"
        )


def optimize_constrained(
    panel: ReturnPanel,
    objective: Objective = Objective.MAX_SHARPE,
    constraints: Constraints | None = None,
    risk_free: pd.Series | float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
    force_sampled: bool = False,
) -> OptimizationResult:
    """Best allocation subject to policy limits.

    With no constraints this delegates to the exact solver where one exists,
    since that is strictly the better answer.

    `force_sampled=True` overrides that and uses the sampled search regardless.
    It exists for `cost_of_constraints`, which must compare like with like: a
    cost computed by differencing an exact solve against a sampled one includes
    the gap between the two methods, and that gap does not shrink as the
    constraint loosens. It shows up as a floor under every reported cost --
    every rule appearing to cost the same amount, including rules that bind on
    nothing.
    """
    constraints = constraints or Constraints()
    constraints.validate(panel.assets)

    if constraints.is_empty and not force_sampled:
        from core.optimize import optimize

        return optimize(
            panel,
            objective,
            risk_free=risk_free,
            tolerance=tolerance,
            n_samples=n_samples,
            seed=seed,
        )

    raw = sample_allocations(len(panel.assets), n_samples, seed=seed)
    feasible = (
        raw
        if constraints.is_empty
        else project_onto_constraints(raw, panel.assets, constraints)
    )

    if len(feasible) == 0:
        raise RuntimeError(
            f"No sampled allocation could be made feasible under: "
            f"{constraints.describe()}. The limits may be jointly impossible "
            f"even though each is individually valid."
        )

    measured = _measure_samples(panel, feasible, risk_free)
    column = _COLUMN_FOR[objective]

    finite = measured[np.isfinite(measured[column])]
    if finite.empty:
        raise RuntimeError(
            f"Every feasible allocation produced a degenerate {column}."
        )

    best_row = (
        finite[column].idxmax()
        if (_MAXIMISE[objective] or objective is Objective.MIN_DRAWDOWN)
        else finite[column].idxmin()
    )
    weights_array = measured.loc[best_row, panel.assets].to_numpy(dtype=float)

    refined = _refine_constrained(
        panel, weights_array, objective, constraints, risk_free, seed
    )
    weights = pd.Series(refined, index=panel.assets)

    if not constraints.satisfied_by(weights):
        raise RuntimeError(
            f"Refinement produced an infeasible allocation: "
            f"{constraints.violations(weights)}"
        )

    stats = portfolio_stats(panel, weights, risk_free=risk_free)
    best_value = score(stats, objective)

    return OptimizationResult(
        weights=weights,
        stats=stats,
        objective=objective,
        method=Method.SAMPLED,
        near_optimal=_near_optimal(measured, objective, best_value, tolerance),
        tolerance=tolerance,
        n_samples=len(feasible),
    )


def _refine_constrained(
    panel: ReturnPanel,
    start: np.ndarray,
    objective: Objective,
    constraints: Constraints,
    risk_free: pd.Series | float | None,
    seed: int,
    rounds: int = 4,
    per_round: int = 400,
) -> np.ndarray:
    """Local search that stays inside the feasible region."""
    rng = np.random.default_rng(seed + 1)
    n = len(start)

    best = start.copy()
    best_value = score(
        portfolio_stats(panel, pd.Series(best, index=panel.assets), risk_free=risk_free),
        objective,
    )

    step = 0.08
    for _ in range(rounds):
        perturbed = best + rng.normal(0.0, step, size=(per_round, n))
        perturbed = np.clip(perturbed, 0.0, None)
        totals = perturbed.sum(axis=1, keepdims=True)
        perturbed = np.divide(
            perturbed, totals, out=np.zeros_like(perturbed), where=totals > 0
        )
        perturbed = project_onto_constraints(perturbed, panel.assets, constraints)
        if len(perturbed) == 0:
            step /= 3.0
            continue

        measured = _measure_samples(panel, perturbed, risk_free)
        column = _COLUMN_FOR[objective]
        finite = measured[np.isfinite(measured[column])]
        if finite.empty:
            step /= 3.0
            continue

        candidate_row = (
            finite[column].idxmax()
            if (_MAXIMISE[objective] or objective is Objective.MIN_DRAWDOWN)
            else finite[column].idxmin()
        )
        candidate_value = float(finite.loc[candidate_row, column])

        better = (
            candidate_value > best_value
            if (_MAXIMISE[objective] or objective is Objective.MIN_DRAWDOWN)
            else candidate_value < best_value
        )
        if better:
            best = finite.loc[candidate_row, panel.assets].to_numpy(dtype=float)
            best_value = candidate_value

        step /= 3.0

    return best


def cost_of_constraints(
    panel: ReturnPanel,
    objective: Objective = Objective.MAX_SHARPE,
    constraints: Constraints | None = None,
    risk_free: pd.Series | float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> ConstraintCost:
    """Solve with and without the constraints, and report the gap.

    Both solves use the same method and the same sample budget, so the
    difference reflects the constraints rather than a change of solver.
    """
    constraints = constraints or Constraints()
    constraints.validate(panel.assets)

    # Both solves use the sampled search, from the same seed and budget, so the
    # difference reflects the constraints rather than a change of method.
    free = optimize_constrained(
        panel,
        objective,
        Constraints(),
        risk_free,
        tolerance,
        n_samples,
        seed,
        force_sampled=True,
    )
    bound = optimize_constrained(
        panel, objective, constraints, risk_free, tolerance, n_samples, seed
    )

    # Any allocation feasible under the constraints is also feasible without
    # them, so the unconstrained answer cannot legitimately be worse. When the
    # search says otherwise it is an artefact of sampling density: projecting
    # into a smaller region concentrates the samples there, so the constrained
    # search covers its space better than the unconstrained one covers a larger
    # space. Adopting the better allocation removes the artefact without
    # pretending the constraint helped.
    if is_better(bound.value, free.value, objective):
        free = OptimizationResult(
            weights=bound.weights,
            stats=bound.stats,
            objective=objective,
            method=bound.method,
            near_optimal=free.near_optimal,
            tolerance=tolerance,
            n_samples=free.n_samples,
        )

    return ConstraintCost(
        objective=objective,
        unconstrained=free,
        constrained=bound,
        constraints=constraints,
    )


def cost_per_constraint(
    panel: ReturnPanel,
    objective: Objective,
    constraints: Constraints,
    risk_free: pd.Series | float | None = None,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """What each individual limit costs, isolated from the others.

    Each row removes one constraint and re-solves, so the reported figure is
    the marginal cost of that rule given the others. Costs do not sum to the
    total -- constraints interact, and two rules can be individually cheap and
    jointly expensive.
    """
    full = optimize_constrained(
        panel,
        objective,
        constraints,
        risk_free,
        n_samples=n_samples,
        seed=seed,
        force_sampled=True,
    )

    rows = []
    for asset, cap in constraints.caps.items():
        without = Constraints(
            caps={k: v for k, v in constraints.caps.items() if k != asset},
            floors=constraints.floors,
            groups=constraints.groups,
        )
        rows.append(("cap", f"{asset} <= {cap:.0%}", without))

    for asset, floor in constraints.floors.items():
        without = Constraints(
            caps=constraints.caps,
            floors={k: v for k, v in constraints.floors.items() if k != asset},
            groups=constraints.groups,
        )
        rows.append(("floor", f"{asset} >= {floor:.0%}", without))

    for group in constraints.groups:
        without = Constraints(
            caps=constraints.caps,
            floors=constraints.floors,
            groups=tuple(g for g in constraints.groups if g is not group),
        )
        rows.append(("group", group.name, without))

    records = []
    for kind, label, relaxed in rows:
        result = optimize_constrained(
            panel,
            objective,
            relaxed,
            risk_free,
            n_samples=n_samples,
            seed=seed,
            force_sampled=True,
        )
        records.append(
            {
                "kind": kind,
                "constraint": label,
                "objective_without": result.value,
                "objective_with": full.value,
                "return_without": result.stats.realised_return,
                "return_with": full.stats.realised_return,
                "cost_bps": (
                    result.stats.realised_return - full.stats.realised_return
                )
                * 10_000,
            }
        )

    return pd.DataFrame(records)
