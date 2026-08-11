"""Tests for scripts/run_optimize.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core.config import Objective  # noqa: E402
from core.optimize import Method, optimize_all  # noqa: E402
from core.panels import ReturnPanel  # noqa: E402
from core.sleeve import build_sleeve  # noqa: E402


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


def test_optimizing_the_sleeved_panel_produces_five_buckets():
    """The optimizer must see the sleeve, not its two components -- otherwise
    the user is allocating to something the slider already decided."""
    sleeved = build_sleeve(_panel(), 0.5)
    results = optimize_all(sleeved, n_samples=1_000)

    for result in results.values():
        assert set(result.weights.index) == set(sleeved.assets)
        assert "gold" not in result.weights.index


def test_methods_are_reported_per_objective():
    sleeved = build_sleeve(_panel(), 0.5)
    results = optimize_all(sleeved, n_samples=1_000)

    assert results[Objective.MIN_VOLATILITY].method is Method.EXACT
    assert results[Objective.MAX_SHARPE].method is Method.EXACT
    assert results[Objective.MAX_SORTINO].method is Method.SAMPLED
    assert results[Objective.MIN_DRAWDOWN].method is Method.SAMPLED


def test_moving_the_slider_changes_the_optimal_allocation():
    """If it did not, the slider would be cosmetic."""
    panel = _panel()
    gold_heavy = optimize_all(build_sleeve(panel, 0.9), n_samples=1_000)
    ex_gold_heavy = optimize_all(build_sleeve(panel, 0.1), n_samples=1_000)

    a = gold_heavy[Objective.MAX_SHARPE].weights["commodities"]
    b = ex_gold_heavy[Objective.MAX_SHARPE].weights["commodities"]
    assert abs(a - b) > 0.01


def test_every_objective_produces_a_valid_allocation():
    sleeved = build_sleeve(_panel(), 0.5)
    for result in optimize_all(sleeved, n_samples=1_000).values():
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert (result.weights >= -1e-9).all()
