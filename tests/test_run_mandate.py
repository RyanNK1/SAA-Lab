"""Tests for scripts/run_mandate.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core.constraints import Constraints, GroupLimit  # noqa: E402
from core.mandate import Mandate, frontier_of_mandates, solve_mandate  # noqa: E402
from core.panels import ReturnPanel  # noqa: E402
from core.sleeve import build_sleeve  # noqa: E402
from run_mandate import GROWTH  # noqa: E402


def _panel(n: int = 245, seed: int = 3) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "equity": rng.normal(0.0065, 0.046, n),
            "fixed_income": rng.normal(0.0025, 0.013, n),
            "private_equity": rng.normal(0.0070, 0.058, n),
            "gold": rng.normal(0.0080, 0.050, n),
            "commodities_ex_gold": rng.normal(0.0020, 0.054, n),
            "cash": np.abs(rng.normal(0.0012, 0.0004, n)),
        },
        index=pd.date_range("2006-03-31", periods=n, freq="ME"),
    )
    frame.iloc[18:35, [0, 2]] -= 0.06
    return ReturnPanel(frame)


def _constraints() -> Constraints:
    return Constraints(
        caps={"private_equity": 0.20},
        floors={"cash": 0.05},
        groups=(GroupLimit("growth", GROWTH, maximum=0.60),),
    )


def test_growth_group_names_assets_present_after_the_sleeve_is_built():
    sleeved = build_sleeve(_panel(), 0.5)
    for asset in GROWTH:
        assert asset in sleeved.assets


def test_the_default_style_mandate_is_solvable():
    sleeved = build_sleeve(_panel(), 0.5)
    mandate = Mandate(
        target_return=0.04, max_volatility=0.10, constraints=_constraints()
    )
    result = solve_mandate(
        sleeved, mandate, risk_free=sleeved.returns["cash"], n_samples=2_000
    )
    assert result.feasible
    assert result.n_qualifying > 1


def test_a_demanding_mandate_is_diagnosed():
    """High return inside a tight budget should be impossible, and the tool
    should say what would have to give rather than only that it failed."""
    sleeved = build_sleeve(_panel(), 0.5)
    mandate = Mandate(
        target_return=0.20, max_volatility=0.05, constraints=_constraints()
    )
    result = solve_mandate(
        sleeved, mandate, risk_free=sleeved.returns["cash"], n_samples=2_000
    )
    assert not result.feasible
    assert result.relaxations
    assert "->" in result.explain()


def test_the_sweep_shows_feasibility_collapsing_as_the_target_rises():
    """The useful shape: many allocations qualify at a low target, fewer as it
    rises, none past some point."""
    sleeved = build_sleeve(_panel(), 0.5)
    frontier = frontier_of_mandates(
        sleeved,
        targets=[0.02, 0.04, 0.06, 0.15],
        max_volatility=0.08,
        constraints=_constraints(),
        risk_free=sleeved.returns["cash"],
        n_samples=2_000,
    )

    assert frontier["feasible"].iloc[0]
    assert not frontier["feasible"].iloc[-1]

    reachable = frontier[frontier["feasible"]]
    assert reachable["n_qualifying"].is_monotonic_decreasing


def test_every_ranking_column_is_usable_on_a_real_result():
    from core.mandate import RANKABLE

    sleeved = build_sleeve(_panel(), 0.5)
    result = solve_mandate(
        sleeved,
        Mandate(target_return=0.03, max_volatility=0.12, constraints=_constraints()),
        risk_free=sleeved.returns["cash"],
        n_samples=2_000,
    )
    for column in RANKABLE:
        ordered = result.ranked(column, limit=5)
        assert len(ordered) == 5
        assert ordered[column].notna().all()
