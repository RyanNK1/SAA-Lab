"""Tests for core/periods.py.

The anchor is `test_a_periods_own_allocation_wins_in_that_period`. An
allocation chosen for a window, knowing what happened in it, must score best
there. If it did not, the optimizer would be failing. That the advantage
usually vanishes elsewhere is the point the module exists to make.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Objective
from core.constraints import Constraints
from core.panels import ReturnPanel
from core.periods import (
    NAMED_REGIMES,
    Period,
    compare_periods,
    consensus_allocation,
    cross_period_performance,
    hindsight_premium,
    resolve_periods,
    rolling_periods,
    weight_stability,
)

FAST = 1_500


def _panel(n: int = 240, seed: int = 7) -> ReturnPanel:
    """A panel with two deliberately different halves, so periods disagree.

    Equity is strong then weak; gold is weak then strong. Bonds and cash are
    constructed so their two halves are *identical* rather than merely drawn
    from the same distribution -- a random draw can easily hand one half a
    materially different mean, and a low-volatility asset's optimal weight is
    very sensitive to that. Relying on "same distribution" to mean "same
    behaviour" is how a control ends up moving more than the variable.
    """
    rng = np.random.default_rng(seed)
    half = n // 2

    equity = np.concatenate(
        [rng.normal(0.010, 0.030, half), rng.normal(0.001, 0.060, n - half)]
    )
    gold = np.concatenate(
        [rng.normal(0.001, 0.050, half), rng.normal(0.012, 0.040, n - half)]
    )

    # Tiled, so both halves are the same series and neither regime favours them.
    stable_bonds = rng.normal(0.002, 0.013, half)
    stable_cash = np.abs(rng.normal(0.0012, 0.0004, half))

    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": equity,
                "bonds": np.tile(stable_bonds, 2)[:n],
                "gold": gold,
                "cash": np.tile(stable_cash, 2)[:n],
            },
            index=pd.date_range("2006-03-31", periods=n, freq="ME"),
        )
    )


def _halves(panel: ReturnPanel) -> list[Period]:
    midpoint = panel.returns.index[len(panel) // 2]
    return [
        Period("first", panel.start, midpoint),
        Period("second", midpoint, panel.end),
    ]


# ---------------------------------------------------------------------------
# Period resolution
# ---------------------------------------------------------------------------

def test_named_regimes_are_ordered_and_non_overlapping():
    """Overlapping regimes would double-count months and make the comparison
    harder to reason about."""
    previous_end = None
    for _, start, end in NAMED_REGIMES:
        assert pd.Timestamp(start) < pd.Timestamp(end)
        if previous_end is not None:
            assert pd.Timestamp(start) >= previous_end
        previous_end = pd.Timestamp(end)


def test_regimes_are_clipped_to_the_panel():
    panel = _panel()
    for period in resolve_periods(panel):
        assert period.start >= panel.start
        assert period.end <= panel.end


def test_regimes_with_too_little_data_are_dropped():
    """A regime represented by four months is not a regime."""
    panel = _panel()
    periods = resolve_periods(panel, min_observations=200)
    assert len(periods) <= 1


def test_regimes_outside_the_sample_are_dropped():
    panel = _panel(n=36)  # ends around 2009
    labels = {p.label for p in resolve_periods(panel)}
    assert "Inflation shock" not in labels


def test_rolling_windows_are_the_requested_length():
    panel = _panel()
    for period in rolling_periods(panel, years=5):
        assert 4.8 < period.years < 5.2


def test_rolling_windows_advance_by_the_step():
    panel = _panel()
    yearly = rolling_periods(panel, years=5, step_months=12)
    half_yearly = rolling_periods(panel, years=5, step_months=6)
    assert len(half_yearly) > len(yearly)


def test_rolling_windows_stay_inside_the_sample():
    panel = _panel()
    for period in rolling_periods(panel, years=5):
        assert period.end <= panel.end


def test_rolling_rejects_nonsense_arguments():
    panel = _panel()
    with pytest.raises(ValueError, match="years"):
        rolling_periods(panel, years=0)
    with pytest.raises(ValueError, match="step_months"):
        rolling_periods(panel, step_months=0)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def test_comparison_returns_one_row_per_period():
    panel = _panel()
    periods = _halves(panel)
    table, results = compare_periods(
        panel, periods, n_samples=FAST
    )
    assert len(table) == len(periods)
    assert set(results) == {p.label for p in periods}


def test_every_period_produces_a_valid_allocation():
    panel = _panel()
    table, _ = compare_periods(panel, _halves(panel), n_samples=FAST)
    assert np.allclose(table[panel.assets].sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    assert (table[panel.assets].to_numpy() >= -1e-9).all()


def test_different_periods_give_different_answers():
    """If they did not, comparing periods would be pointless. The fixture has
    two deliberately different halves."""
    panel = _panel()
    table, _ = compare_periods(panel, _halves(panel), n_samples=FAST)
    difference = (table.loc["first", panel.assets] - table.loc["second", panel.assets]).abs()
    assert difference.max() > 0.10


def test_constraints_are_applied_in_every_period():
    panel = _panel()
    constraints = Constraints(floors={"cash": 0.10}, caps={"equity": 0.40})
    _, results = compare_periods(
        panel, _halves(panel), constraints=constraints, n_samples=FAST
    )
    for result in results.values():
        assert constraints.satisfied_by(result.weights)


def test_comparison_rejects_periods_outside_the_panel():
    panel = _panel()
    outside = [Period("far future", pd.Timestamp("2090-01-01"), pd.Timestamp("2095-01-01"))]
    with pytest.raises((ValueError, KeyError)):
        compare_periods(panel, outside, n_samples=FAST)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def test_stability_reports_the_spread_per_asset():
    panel = _panel()
    table, _ = compare_periods(panel, _halves(panel), n_samples=FAST)
    stability = weight_stability(table, panel.assets)

    assert set(stability.index) == set(panel.assets)
    assert (stability["spread"] >= 0).all()
    assert (stability["max"] >= stability["min"]).all()


def test_stability_is_sorted_by_how_much_the_weight_moves():
    panel = _panel()
    table, _ = compare_periods(panel, _halves(panel), n_samples=FAST)
    spreads = weight_stability(table, panel.assets)["spread"].to_numpy()
    assert (np.diff(spreads) <= 1e-12).all()


def test_an_asset_that_reverses_moves_more_than_one_that_does_not():
    """Gold is deliberately poor in the first half and strong in the second;
    bonds are the same throughout. Gold's optimal weight must therefore be the
    less stable of the two.

    Stated as a comparison rather than against a fixed threshold. An absolute
    cutoff would be a claim about this fixture's random draw, not about the
    behaviour being tested, and would need retuning whenever the seed changed.
    """
    panel = _panel()
    table, _ = compare_periods(panel, _halves(panel), n_samples=FAST)
    stability = weight_stability(table, panel.assets)

    assert stability.loc["gold", "spread"] > stability.loc["bonds", "spread"]
    assert stability.loc["gold", "spread"] > 0.02, "should move noticeably"


# ---------------------------------------------------------------------------
# Cross-period performance
# ---------------------------------------------------------------------------

def test_a_periods_own_allocation_wins_in_that_period():
    """In-sample by construction: it was chosen knowing what happened. If it
    lost, the optimizer would be failing."""
    panel = _panel()
    periods = _halves(panel)
    _, results = compare_periods(panel, periods, n_samples=FAST)
    matrix = cross_period_performance(panel, results, periods)

    for label in matrix.columns:
        assert matrix.loc[label, label] >= matrix[label].max() - 1e-6


def test_the_matrix_is_square_and_complete():
    panel = _panel()
    periods = _halves(panel)
    _, results = compare_periods(panel, periods, n_samples=FAST)
    matrix = cross_period_performance(panel, results, periods)

    assert matrix.shape == (len(periods), len(periods))
    assert matrix.notna().all().all()


def test_hindsight_premium_is_non_negative():
    """The in-sample winner cannot do worse than the average of the others in
    its own period -- it beat all of them there."""
    panel = _panel()
    periods = _halves(panel)
    _, results = compare_periods(panel, periods, n_samples=FAST)
    premium = hindsight_premium(cross_period_performance(panel, results, periods))

    assert (premium["premium"] >= -1e-6).all()


def test_hindsight_premium_is_material_when_regimes_differ():
    """The point of the whole module: knowing the answer in advance is worth a
    lot, and nobody gets to."""
    panel = _panel()
    periods = _halves(panel)
    _, results = compare_periods(panel, periods, n_samples=FAST)
    premium = hindsight_premium(cross_period_performance(panel, results, periods))

    assert premium["premium"].max() > 0.05


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

def test_consensus_is_a_valid_allocation():
    panel = _panel()
    table, _ = compare_periods(panel, _halves(panel), n_samples=FAST)
    consensus = consensus_allocation(table, panel.assets)

    assert consensus.sum() == pytest.approx(1.0)
    assert (consensus >= -1e-12).all()


def test_consensus_sits_between_the_period_answers():
    panel = _panel()
    table, _ = compare_periods(panel, _halves(panel), n_samples=FAST)
    consensus = consensus_allocation(table, panel.assets)

    for asset in panel.assets:
        assert table[asset].min() - 1e-9 <= consensus[asset]
        assert consensus[asset] <= table[asset].max() + 1e-9


def test_consensus_weights_longer_periods_more_heavily():
    """A three-month window should not count as much as a seven-year one."""
    assets = ["a", "b"]
    table = pd.DataFrame(
        {"a": [1.0, 0.0], "b": [0.0, 1.0], "months": [120, 12]},
        index=["long", "short"],
    )
    weighted = consensus_allocation(table, assets, weight_by_months=True)
    unweighted = consensus_allocation(table, assets, weight_by_months=False)

    assert weighted["a"] > unweighted["a"]
    assert weighted["a"] == pytest.approx(120 / 132)
