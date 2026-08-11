"""Cross-module consistency checks.

Each module is tested on its own elsewhere. These check that they compose
correctly -- the class of bug that unit tests miss because every piece is
individually right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.panels import ReturnPanel
from core.portfolio import portfolio_stats, risk_contributions
from core.sleeve import build_sleeve


def _panel(n: int = 180, seed: int = 7) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": rng.normal(0.006, 0.045, n),
                "fixed_income": rng.normal(0.002, 0.013, n),
                "gold": rng.normal(0.007, 0.050, n),
                "commodities_ex_gold": rng.normal(0.002, 0.054, n),
                "cash": np.abs(rng.normal(0.0012, 0.0004, n)),
            },
            index=pd.date_range("2006-03-31", periods=n, freq="ME"),
        )
    )


def test_sleeve_then_measure_equals_the_equivalent_expanded_portfolio():
    """Building a 60/40 commodities sleeve and allocating 25% to it must give
    exactly the same portfolio as allocating 15% gold and 10% ex-gold
    directly. If these diverge, the sleeve is not a pass-through and the
    slider is quietly changing the allocation."""
    panel = _panel()
    gold_weight, sleeve_weight = 0.6, 0.25

    sleeved = build_sleeve(panel, gold_weight)
    via_sleeve = portfolio_stats(
        sleeved,
        {"equity": 0.5, "fixed_income": 0.2, "commodities": sleeve_weight, "cash": 0.05},
        risk_free=panel.returns["cash"],
    )
    direct = portfolio_stats(
        panel,
        {
            "equity": 0.5,
            "fixed_income": 0.2,
            "gold": sleeve_weight * gold_weight,
            "commodities_ex_gold": sleeve_weight * (1 - gold_weight),
            "cash": 0.05,
        },
    )

    assert via_sleeve.volatility == pytest.approx(direct.volatility, rel=1e-12)
    assert via_sleeve.realised_return == pytest.approx(direct.realised_return, rel=1e-12)
    assert via_sleeve.max_drawdown == pytest.approx(direct.max_drawdown, rel=1e-12)
    assert via_sleeve.sharpe == pytest.approx(direct.sharpe, rel=1e-9)


def test_marginal_risk_matches_a_numerical_derivative():
    """Marginal risk contribution is defined as the partial derivative of
    portfolio volatility with respect to each weight. Checked against a finite
    difference, which is independent of the closed-form implementation."""
    panel = _panel()
    weights = {
        "equity": 0.4,
        "fixed_income": 0.2,
        "gold": 0.2,
        "commodities_ex_gold": 0.15,
        "cash": 0.05,
    }
    rc = risk_contributions(panel, weights)

    cov = panel.ann_cov().to_numpy()
    w = np.array([weights[a] for a in panel.assets])
    eps = 1e-6

    for asset in panel.assets:
        j = panel.assets.index(asset)
        bumped = w.copy()
        bumped[j] += eps
        numerical = (np.sqrt(bumped @ cov @ bumped) - np.sqrt(w @ cov @ w)) / eps
        assert rc.loc[asset, "marginal_risk"] == pytest.approx(numerical, abs=1e-6)


def test_slider_endpoints_reproduce_single_component_portfolios():
    """A sleeve at 100% gold, allocated 25%, must equal a direct 25% gold
    allocation."""
    panel = _panel()
    sleeved = build_sleeve(panel, 1.0)

    via_sleeve = portfolio_stats(
        sleeved,
        {"equity": 0.5, "fixed_income": 0.2, "commodities": 0.25, "cash": 0.05},
        risk_free=panel.returns["cash"],
    )
    direct = portfolio_stats(
        panel,
        {
            "equity": 0.5,
            "fixed_income": 0.2,
            "gold": 0.25,
            "commodities_ex_gold": 0.0,
            "cash": 0.05,
        },
    )
    assert via_sleeve.volatility == pytest.approx(direct.volatility, rel=1e-12)


def test_selecting_a_subset_then_measuring_is_unaffected_by_dropped_assets():
    """Deselecting an asset must not change the statistics of a portfolio that
    never held it."""
    panel = _panel()
    weights = {"equity": 0.7, "fixed_income": 0.3}

    subset = panel.select(["equity", "fixed_income"])
    full = panel.select(["equity", "fixed_income", "gold", "cash"])

    a = portfolio_stats(subset, weights, risk_free=0.0)
    b = portfolio_stats(full, {**weights, "gold": 0.0, "cash": 0.0}, risk_free=0.0)

    assert a.volatility == pytest.approx(b.volatility, rel=1e-12)
    assert a.max_drawdown == pytest.approx(b.max_drawdown, rel=1e-12)


def test_period_slicing_then_measuring_uses_only_that_period():
    """Statistics for a sub-period must match measuring a panel built from
    only those months -- no leakage from surrounding data."""
    panel = _panel()
    weights = {
        "equity": 0.5,
        "fixed_income": 0.2,
        "gold": 0.15,
        "commodities_ex_gold": 0.1,
        "cash": 0.05,
    }

    sliced = panel.between("2010-01-01", "2015-12-31")
    rebuilt = ReturnPanel(
        panel.returns.loc["2010-01-01":"2015-12-31"], panel.periods_per_year
    )

    a = portfolio_stats(sliced, weights)
    b = portfolio_stats(rebuilt, weights)

    assert a.volatility == pytest.approx(b.volatility, rel=1e-12)
    assert a.realised_return == pytest.approx(b.realised_return, rel=1e-12)
    assert a.n_periods == b.n_periods
