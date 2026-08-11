"""Tests for core/constraints.py.

The anchor is `test_constraints_never_improve_the_objective`. Constraints
shrink the feasible set, so a constrained optimum can never beat an
unconstrained one. A negative cost means the optimizer failed, not that the
rule helped -- and since the whole module exists to report that cost, the
identity has to hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Objective
from core.constraints import (
    ConstraintCost,
    Constraints,
    GroupLimit,
    cost_of_constraints,
    cost_per_constraint,
    optimize_constrained,
    project_onto_constraints,
)
from core.panels import ReturnPanel

FAST = 2_000


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2006-03-31", periods=n, freq="ME")


def _panel(n: int = 180, seed: int = 11) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": rng.normal(0.006, 0.046, n),
                "fixed_income": rng.normal(0.002, 0.013, n),
                "private_equity": rng.normal(0.006, 0.058, n),
                "commodities": rng.normal(0.004, 0.042, n),
                "cash": np.abs(rng.normal(0.0012, 0.0004, n)),
            },
            index=_dates(n),
        )
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_floors_summing_above_one_are_rejected():
    constraints = Constraints(floors={"equity": 0.6, "fixed_income": 0.5})
    with pytest.raises(ValueError, match="more than the whole portfolio"):
        constraints.validate(_panel().assets)


def test_caps_summing_below_one_are_rejected():
    """If everything is capped, the portfolio cannot be fully invested."""
    constraints = Constraints(
        caps={a: 0.1 for a in _panel().assets}
    )
    with pytest.raises(ValueError, match="cannot be fully invested"):
        constraints.validate(_panel().assets)


def test_a_floor_above_its_own_cap_is_rejected():
    with pytest.raises(ValueError, match="exceeds its cap"):
        Constraints(caps={"equity": 0.2}, floors={"equity": 0.5})


def test_unknown_assets_are_named():
    constraints = Constraints(caps={"crypto": 0.1})
    with pytest.raises(KeyError, match="crypto"):
        constraints.validate(_panel().assets)


def test_group_needing_more_than_its_assets_can_supply_is_rejected():
    constraints = Constraints(
        caps={"equity": 0.1, "private_equity": 0.1},
        groups=(GroupLimit("growth", ("equity", "private_equity"), minimum=0.5),),
    )
    with pytest.raises(ValueError, match="capped at"):
        constraints.validate(_panel().assets)


def test_group_with_neither_bound_is_rejected():
    with pytest.raises(ValueError, match="neither a floor nor a cap"):
        GroupLimit("growth", ("equity",))


def test_group_floor_above_its_own_cap_is_rejected():
    with pytest.raises(ValueError, match="exceeds its cap"):
        GroupLimit("growth", ("equity",), maximum=0.3, minimum=0.5)


# ---------------------------------------------------------------------------
# Satisfaction checks
# ---------------------------------------------------------------------------

def test_satisfied_by_detects_a_breached_cap():
    constraints = Constraints(caps={"equity": 0.3})
    weights = pd.Series({"equity": 0.5, "cash": 0.5})
    assert not constraints.satisfied_by(weights)
    assert "exceeds its 30.0% cap" in constraints.violations(weights)[0]


def test_satisfied_by_detects_a_breached_floor():
    constraints = Constraints(floors={"cash": 0.10})
    weights = pd.Series({"equity": 0.95, "cash": 0.05})
    assert not constraints.satisfied_by(weights)
    assert "below its 10.0% floor" in constraints.violations(weights)[0]


def test_satisfied_by_detects_a_breached_group():
    constraints = Constraints(
        groups=(GroupLimit("growth", ("equity", "private_equity"), maximum=0.5),)
    )
    weights = pd.Series({"equity": 0.4, "private_equity": 0.3, "cash": 0.3})
    assert not constraints.satisfied_by(weights)
    assert "growth" in constraints.violations(weights)[0]


def test_a_satisfying_allocation_reports_no_violations():
    constraints = Constraints(caps={"equity": 0.5}, floors={"cash": 0.05})
    weights = pd.Series({"equity": 0.4, "cash": 0.6})
    assert constraints.satisfied_by(weights)
    assert constraints.violations(weights) == []


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def test_projection_produces_feasible_allocations():
    panel = _panel()
    constraints = Constraints(
        caps={"private_equity": 0.20},
        floors={"cash": 0.05, "fixed_income": 0.10},
        groups=(GroupLimit("growth", ("equity", "private_equity"), maximum=0.60),),
    )
    rng = np.random.default_rng(2)
    raw = rng.dirichlet(np.ones(len(panel.assets)), size=800)

    projected = project_onto_constraints(raw, panel.assets, constraints)

    assert len(projected) > 100, "projection should not discard almost everything"
    assert np.allclose(projected.sum(axis=1), 1.0, atol=1e-8)
    for row in projected:
        assert constraints.satisfied_by(pd.Series(row, index=panel.assets))


def test_projection_leaves_already_feasible_allocations_summing_to_one():
    panel = _panel()
    constraints = Constraints(caps={"equity": 0.9})
    rng = np.random.default_rng(5)
    raw = rng.dirichlet(np.ones(len(panel.assets)), size=200)

    projected = project_onto_constraints(raw, panel.assets, constraints)
    assert np.allclose(projected.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Constrained optimisation
# ---------------------------------------------------------------------------

def test_constrained_result_respects_every_limit():
    panel = _panel()
    constraints = Constraints(
        caps={"private_equity": 0.15},
        floors={"cash": 0.05},
        groups=(GroupLimit("growth", ("equity", "private_equity"), maximum=0.55),),
    )
    for objective in Objective:
        result = optimize_constrained(
            panel, objective, constraints, n_samples=FAST
        )
        assert constraints.satisfied_by(result.weights), constraints.violations(
            result.weights
        )
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_empty_constraints_match_the_unconstrained_optimizer():
    from core.optimize import optimize

    panel = _panel()
    a = optimize_constrained(panel, Objective.MAX_SHARPE, Constraints(), n_samples=FAST)
    b = optimize(panel, Objective.MAX_SHARPE, n_samples=FAST)
    assert a.value == pytest.approx(b.value, rel=1e-9)


def test_a_floor_forces_drawdown_out_of_cash():
    """Unconstrained, minimum drawdown hides in cash -- a correct answer to the
    question asked and useless as an allocation. A floor on risky assets makes
    it choose something a person might actually hold."""
    panel = _panel()
    constraints = Constraints(
        groups=(
            GroupLimit(
                "risky",
                ("equity", "fixed_income", "private_equity", "commodities"),
                minimum=0.60,
            ),
        )
    )
    result = optimize_constrained(
        panel, Objective.MIN_DRAWDOWN, constraints, n_samples=FAST
    )
    assert result.weights["cash"] <= 0.40 + 1e-6
    assert result.stats.max_drawdown < 0.0


# ---------------------------------------------------------------------------
# The cost
# ---------------------------------------------------------------------------

def test_constraints_never_improve_the_objective():
    """The anchor. A smaller feasible set cannot contain a better answer, so
    the cost is non-negative for every objective."""
    panel = _panel()
    constraints = Constraints(
        caps={"private_equity": 0.10, "equity": 0.40},
        floors={"cash": 0.05, "fixed_income": 0.15},
    )
    for objective in Objective:
        cost = cost_of_constraints(panel, objective, constraints, n_samples=FAST)
        assert cost.cost >= -1e-6, f"{objective.value} cost was {cost.cost}"


def test_a_non_binding_constraint_costs_nothing():
    """A cap the optimum was never near must be free."""
    panel = _panel()
    free = optimize_constrained(
        panel, Objective.MAX_SHARPE, Constraints(), n_samples=FAST
    )
    slack = float(free.weights.max()) + 0.10

    if slack >= 1.0:
        pytest.skip("optimum is already concentrated; no slack to test with")

    generous = Constraints(caps={a: min(slack, 1.0) for a in panel.assets})
    cost = cost_of_constraints(
        panel, Objective.MAX_SHARPE, generous, n_samples=FAST
    )
    assert abs(cost.cost) < 0.05


def test_a_severe_constraint_costs_more_than_a_slack_one():
    """Tightening a binding constraint should cost more.

    Asserted between clearly different regimes rather than between adjacent
    ones. Exact monotonicity in constraint tightness cannot be guaranteed by a
    sampled search: projecting into a smaller feasible region concentrates the
    samples there, so a tighter constraint is sometimes searched *better* than
    a looser one and the measured cost dips. That is a limitation of the
    method, not of the constraint, and pretending otherwise would mean tuning
    the test until it passed.
    """
    panel = _panel()
    slack = cost_of_constraints(
        panel,
        Objective.MAX_SHARPE,
        Constraints(caps={"fixed_income": 0.95}),
        n_samples=FAST,
    )
    severe = cost_of_constraints(
        panel,
        Objective.MAX_SHARPE,
        Constraints(floors={"private_equity": 0.40}),
        n_samples=FAST,
    )
    assert severe.cost > slack.cost + 0.01


def test_capping_cash_does_not_change_maximum_sharpe():
    """Not a bug, and worth pinning down so it is not mistaken for one.

    Mixing cash into a risky portfolio leaves Sharpe unchanged: cash adds no
    excess return and scales volatility proportionally, so the ratio survives.
    A cap on cash therefore costs nothing on this objective -- though it very
    much changes volatility and drawdown.
    """
    panel = _panel()
    cost = cost_of_constraints(
        panel,
        Objective.MAX_SHARPE,
        Constraints(caps={"cash": 0.10}),
        n_samples=FAST,
    )
    assert abs(cost.cost) < 0.01

    volatility_cost = cost_of_constraints(
        panel,
        Objective.MIN_VOLATILITY,
        Constraints(caps={"cash": 0.10}),
        n_samples=FAST,
    )
    assert volatility_cost.cost > 0.01, "it must still cost something on risk"


def test_cost_is_never_negative_across_many_constraint_sets():
    """The identity that must always hold: a smaller feasible set cannot
    contain a better answer."""
    panel = _panel()
    sets = [
        Constraints(caps={"fixed_income": cap}) for cap in (0.2, 0.4, 0.6, 0.8)
    ] + [
        Constraints(floors={"cash": floor}) for floor in (0.05, 0.15, 0.30)
    ]
    for constraints in sets:
        for objective in (Objective.MAX_SHARPE, Objective.MIN_VOLATILITY):
            cost = cost_of_constraints(
                panel, objective, constraints, n_samples=FAST
            )
            assert cost.cost >= -1e-9, f"{constraints.describe()} / {objective}"


def test_cost_is_reported_in_basis_points():
    panel = _panel()
    cost = cost_of_constraints(
        panel,
        Objective.MAX_SHARPE,
        Constraints(caps={"fixed_income": 0.20}),
        n_samples=FAST,
    )
    assert "bps a year" in cost.describe()


def test_empty_constraints_cost_nothing():
    panel = _panel()
    cost = cost_of_constraints(
        panel, Objective.MAX_SHARPE, Constraints(), n_samples=FAST
    )
    assert abs(cost.cost) < 1e-9
    assert cost.describe() == "no constraints applied"


def test_per_constraint_costs_are_reported_individually():
    panel = _panel()
    constraints = Constraints(
        caps={"private_equity": 0.10},
        floors={"cash": 0.10},
        groups=(GroupLimit("growth", ("equity", "private_equity"), maximum=0.50),),
    )
    table = cost_per_constraint(
        panel, Objective.MAX_SHARPE, constraints, n_samples=FAST
    )

    assert len(table) == 3
    assert set(table["kind"]) == {"cap", "floor", "group"}
    assert "cost_bps" in table.columns


def test_infeasible_sampling_fails_loudly():
    """Limits that are individually valid can be jointly impossible. Better a
    clear failure than an answer that quietly breaks a rule."""
    panel = _panel()
    constraints = Constraints(
        floors={"cash": 0.50},
        groups=(
            GroupLimit(
                "risky",
                ("equity", "fixed_income", "private_equity", "commodities"),
                minimum=0.90,
            ),
        ),
    )
    with pytest.raises((RuntimeError, ValueError)):
        optimize_constrained(
            panel, Objective.MAX_SHARPE, constraints, n_samples=500
        )


def test_describe_lists_every_limit():
    constraints = Constraints(
        caps={"private_equity": 0.20},
        floors={"cash": 0.05},
        groups=(GroupLimit("growth", ("equity",), maximum=0.60),),
    )
    text = constraints.describe()
    assert "private_equity <= 20%" in text
    assert "cash >= 5%" in text
    assert "growth <= 60%" in text
