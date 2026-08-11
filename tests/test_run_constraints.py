"""Tests for scripts/run_constraints.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core.config import Objective  # noqa: E402
from core.constraints import (  # noqa: E402
    Constraints,
    GroupLimit,
    cost_of_constraints,
    cost_per_constraint,
)
from core.panels import ReturnPanel  # noqa: E402
from core.sleeve import build_sleeve  # noqa: E402
from run_constraints import GROWTH  # noqa: E402


def _panel(n: int = 180, seed: int = 3) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": rng.normal(0.006, 0.046, n),
                "fixed_income": rng.normal(0.002, 0.013, n),
                "private_equity": rng.normal(0.006, 0.058, n),
                "gold": rng.normal(0.007, 0.050, n),
                "commodities_ex_gold": rng.normal(0.001, 0.054, n),
                "cash": np.abs(rng.normal(0.0012, 0.0004, n)),
            },
            index=pd.date_range("2006-03-31", periods=n, freq="ME"),
        )
    )


def _default_constraints() -> Constraints:
    return Constraints(
        caps={"private_equity": 0.20},
        floors={"cash": 0.05, "fixed_income": 0.15},
        groups=(GroupLimit("growth", GROWTH, maximum=0.60),),
    )


def test_growth_group_names_real_assets():
    sleeved = build_sleeve(_panel(), 0.5)
    for asset in GROWTH:
        assert asset in sleeved.assets


def test_default_constraints_are_feasible_on_the_sleeved_panel():
    sleeved = build_sleeve(_panel(), 0.5)
    _default_constraints().validate(sleeved.assets)


def test_the_reported_cost_is_non_negative():
    sleeved = build_sleeve(_panel(), 0.5)
    cash = sleeved.returns["cash"]
    result = cost_of_constraints(
        sleeved,
        Objective.MAX_SHARPE,
        _default_constraints(),
        risk_free=cash,
        n_samples=2_000,
    )
    assert result.cost >= -1e-9


def test_the_constrained_answer_obeys_every_rule():
    sleeved = build_sleeve(_panel(), 0.5)
    constraints = _default_constraints()
    result = cost_of_constraints(
        sleeved,
        Objective.MAX_SHARPE,
        constraints,
        risk_free=sleeved.returns["cash"],
        n_samples=2_000,
    )
    assert constraints.satisfied_by(result.constrained.weights), constraints.violations(
        result.constrained.weights
    )


def test_isolation_table_has_one_row_per_rule():
    sleeved = build_sleeve(_panel(), 0.5)
    constraints = _default_constraints()
    table = cost_per_constraint(
        sleeved,
        Objective.MAX_SHARPE,
        constraints,
        risk_free=sleeved.returns["cash"],
        n_samples=2_000,
    )
    expected = len(constraints.caps) + len(constraints.floors) + len(constraints.groups)
    assert len(table) == expected


def test_a_risky_floor_pulls_drawdown_out_of_cash():
    """The behaviour the script demonstrates: a floor turns a degenerate
    all-cash answer into an allocation someone might hold."""
    from core.constraints import optimize_constrained

    sleeved = build_sleeve(_panel(), 0.5)
    risky = tuple(a for a in sleeved.assets if a != "cash")
    forced = Constraints(groups=(GroupLimit("risky assets", risky, minimum=0.60),))

    free = optimize_constrained(
        sleeved, Objective.MIN_DRAWDOWN, Constraints(), n_samples=2_000
    )
    bound = optimize_constrained(
        sleeved, Objective.MIN_DRAWDOWN, forced, n_samples=2_000
    )

    assert free.weights["cash"] > 0.9
    assert bound.weights["cash"] <= 0.40 + 1e-6
    assert bound.stats.max_drawdown < free.stats.max_drawdown
