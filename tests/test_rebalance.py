"""Tests for core/rebalance.py.

The anchoring test is `test_monthly_reproduces_the_step_two_measurement`.
Rebalancing every period is what makes the weighted sum of asset returns the
portfolio return -- which is exactly what the measurement layer already
assumes. If the simulator and that layer disagree at monthly, one of them is
wrong, and everything built on either is suspect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Rebalance
from core.panels import ReturnPanel
from core.portfolio import portfolio_stats
from core.rebalance import (
    RebalanceSpec,
    compare_schedules,
    measure_path,
    simulate,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2006-03-31", periods=n, freq="ME")


def _panel(n: int = 240, seed: int = 11) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": rng.normal(0.006, 0.046, n),
                "bonds": rng.normal(0.002, 0.013, n),
                "gold": rng.normal(0.007, 0.050, n),
                "cash": np.abs(rng.normal(0.0012, 0.0004, n)),
            },
            index=_dates(n),
        )
    )


_W = {"equity": 0.5, "bonds": 0.25, "gold": 0.2, "cash": 0.05}


# ---------------------------------------------------------------------------
# The anchor
# ---------------------------------------------------------------------------

def test_monthly_reproduces_the_step_two_measurement():
    """Monthly rebalancing with no costs must match `portfolio_stats` exactly.
    Two independent implementations of the same portfolio."""
    panel = _panel()
    path = simulate(panel, _W, RebalanceSpec(Rebalance.MONTHLY, cost_bps=0.0))
    direct = portfolio_stats(panel, _W)

    assert np.allclose(
        path.returns.to_numpy(),
        (panel.returns * pd.Series(_W)).sum(axis=1).to_numpy(),
        atol=1e-15,
    )

    measured = measure_path(panel, path)
    assert measured.volatility == pytest.approx(direct.volatility, rel=1e-12)
    assert measured.realised_return == pytest.approx(direct.realised_return, rel=1e-12)
    assert measured.max_drawdown == pytest.approx(direct.max_drawdown, rel=1e-12)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def test_drift_is_hand_computable():
    """50/50, one asset returns 10% and the other nothing. The holdings become
    0.55 and 0.50, so weights are 0.55/1.05 and 0.50/1.05."""
    idx = _dates(3)
    panel = ReturnPanel(
        pd.DataFrame({"a": [0.10, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}, index=idx)
    )
    path = simulate(panel, {"a": 0.5, "b": 0.5}, RebalanceSpec(Rebalance.NEVER))

    assert path.weights.loc[idx[1], "a"] == pytest.approx(0.55 / 1.05)
    assert path.weights.loc[idx[1], "b"] == pytest.approx(0.50 / 1.05)


def test_never_rebalancing_lets_the_winner_take_over():
    """The reason the setting matters: a portfolio nobody corrects ends up
    holding whatever grew fastest."""
    idx = _dates(120)
    panel = ReturnPanel(
        pd.DataFrame(
            {"winner": np.full(120, 0.010), "loser": np.full(120, 0.001)}, index=idx
        )
    )
    path = simulate(panel, {"winner": 0.5, "loser": 0.5}, Rebalance.NEVER)

    assert path.final_weights["winner"] > 0.70
    assert path.max_drift > 0.20


def test_rebalancing_holds_the_shape_the_owner_chose():
    idx = _dates(120)
    panel = ReturnPanel(
        pd.DataFrame(
            {"winner": np.full(120, 0.010), "loser": np.full(120, 0.001)}, index=idx
        )
    )
    path = simulate(
        panel, {"winner": 0.5, "loser": 0.5}, RebalanceSpec(Rebalance.MONTHLY, 0.0)
    )
    assert path.max_drift < 0.01


def test_single_asset_never_drifts():
    panel = _panel()
    for schedule in Rebalance:
        path = simulate(
            panel,
            {"equity": 1.0, "bonds": 0.0, "gold": 0.0, "cash": 0.0},
            RebalanceSpec(schedule),
        )
        assert path.max_drift < 1e-12
        assert path.total_turnover == pytest.approx(0.0)


def test_identical_assets_never_drift():
    """Two assets with the same returns keep the same split, so no
    rebalancing is ever required."""
    idx = _dates(60)
    rng = np.random.default_rng(3)
    series = rng.normal(0.006, 0.04, 60)
    panel = ReturnPanel(pd.DataFrame({"a": series, "b": series}, index=idx))

    path = simulate(panel, {"a": 0.5, "b": 0.5}, Rebalance.NEVER)
    assert path.max_drift < 1e-12


def test_weights_stay_valid_throughout():
    panel = _panel()
    for schedule in Rebalance:
        path = simulate(panel, _W, RebalanceSpec(schedule))
        assert np.allclose(path.weights.sum(axis=1).to_numpy(), 1.0, atol=1e-12)
        assert (path.weights.to_numpy() >= -1e-12).all()


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def test_never_trades_nothing():
    panel = _panel()
    path = simulate(panel, _W, Rebalance.NEVER)
    assert path.n_rebalances == 0
    assert path.total_cost == pytest.approx(0.0)


def test_calendar_schedules_trade_at_the_right_frequency():
    panel = _panel(n=240)
    expected = {Rebalance.MONTHLY: 239, Rebalance.QUARTERLY: 79, Rebalance.ANNUAL: 19}

    for schedule, count in expected.items():
        path = simulate(panel, _W, RebalanceSpec(schedule, cost_bps=0.0))
        assert path.n_rebalances == count, f"{schedule.value} traded {path.n_rebalances}"


def test_more_frequent_rebalancing_means_more_turnover():
    panel = _panel()
    turnovers = {
        schedule: simulate(panel, _W, RebalanceSpec(schedule, 0.0)).total_turnover
        for schedule in (
            Rebalance.NEVER,
            Rebalance.ANNUAL,
            Rebalance.QUARTERLY,
            Rebalance.MONTHLY,
        )
    }
    assert (
        turnovers[Rebalance.NEVER]
        < turnovers[Rebalance.ANNUAL]
        < turnovers[Rebalance.QUARTERLY]
        < turnovers[Rebalance.MONTHLY]
    )


def test_first_period_is_never_charged():
    """There is no prior portfolio to trade out of."""
    panel = _panel()
    for schedule in Rebalance:
        path = simulate(panel, _W, RebalanceSpec(schedule))
        assert path.turnover.iloc[0] == 0.0
        assert path.costs.iloc[0] == 0.0


def test_threshold_only_trades_when_drift_exceeds_the_band():
    panel = _panel()
    tight = simulate(panel, _W, RebalanceSpec(Rebalance.THRESHOLD, 0.0, 0.01))
    loose = simulate(panel, _W, RebalanceSpec(Rebalance.THRESHOLD, 0.0, 0.20))

    assert tight.n_rebalances > loose.n_rebalances


def test_threshold_trades_less_than_monthly():
    """Its whole purpose: trade only when it matters."""
    panel = _panel()
    threshold = simulate(panel, _W, RebalanceSpec(Rebalance.THRESHOLD, 0.0, 0.05))
    monthly = simulate(panel, _W, RebalanceSpec(Rebalance.MONTHLY, 0.0))

    assert threshold.total_turnover < monthly.total_turnover


def test_threshold_keeps_drift_within_its_band():
    """Drift is measured before trading, so it may exceed the band within a
    period, but should not run far beyond it."""
    panel = _panel()
    band = 0.05
    path = simulate(panel, _W, RebalanceSpec(Rebalance.THRESHOLD, 0.0, band))
    assert path.max_drift < band * 3


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

def test_costs_reduce_returns():
    panel = _panel()
    free = measure_path(panel, simulate(panel, _W, RebalanceSpec(Rebalance.MONTHLY, 0.0)))
    costly = measure_path(
        panel, simulate(panel, _W, RebalanceSpec(Rebalance.MONTHLY, 50.0))
    )
    assert costly.realised_return < free.realised_return


def test_costs_scale_with_the_rate():
    panel = _panel()
    ten = simulate(panel, _W, RebalanceSpec(Rebalance.ANNUAL, 10.0)).total_cost
    twenty = simulate(panel, _W, RebalanceSpec(Rebalance.ANNUAL, 20.0)).total_cost
    assert twenty == pytest.approx(2 * ten, rel=1e-12)


def test_cost_equals_turnover_times_the_rate():
    panel = _panel()
    path = simulate(panel, _W, RebalanceSpec(Rebalance.QUARTERLY, 25.0))
    assert path.total_cost == pytest.approx(
        path.total_turnover * 25.0 / 10_000.0, rel=1e-12
    )


def test_never_pays_nothing_regardless_of_the_rate():
    panel = _panel()
    path = simulate(panel, _W, RebalanceSpec(Rebalance.NEVER, cost_bps=100.0))
    assert path.total_cost == pytest.approx(0.0)


def test_frequent_rebalancing_can_lose_to_never_once_costs_bite():
    """Rebalancing is not free money. With a strong persistent trend and heavy
    costs, trimming the winner every month is worse than leaving it alone."""
    idx = _dates(120)
    panel = ReturnPanel(
        pd.DataFrame(
            {"winner": np.full(120, 0.012), "loser": np.full(120, 0.0)}, index=idx
        )
    )
    never = measure_path(
        panel, simulate(panel, {"winner": 0.5, "loser": 0.5}, Rebalance.NEVER)
    )
    monthly = measure_path(
        panel,
        simulate(
            panel, {"winner": 0.5, "loser": 0.5}, RebalanceSpec(Rebalance.MONTHLY, 50.0)
        ),
    )
    assert never.realised_return > monthly.realised_return


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------

def test_spec_rejects_negative_costs():
    with pytest.raises(ValueError, match="cost_bps"):
        RebalanceSpec(Rebalance.MONTHLY, cost_bps=-1.0)


def test_spec_rejects_an_impossible_band():
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="threshold_band"):
            RebalanceSpec(Rebalance.THRESHOLD, threshold_band=bad)


def test_spec_rejects_a_string_schedule():
    with pytest.raises(TypeError, match="Rebalance member"):
        RebalanceSpec("monthly")  # type: ignore[arg-type]


def test_bad_weights_are_rejected_before_simulating():
    panel = _panel()
    with pytest.raises(ValueError, match="sum to 1.0"):
        simulate(panel, {"equity": 0.5, "bonds": 0.2, "gold": 0.2, "cash": 0.2})


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def test_comparison_covers_every_schedule():
    panel = _panel()
    table = compare_schedules(panel, _W)
    assert set(table.index) == {s.value for s in Rebalance}


def test_comparison_reports_turnover_and_cost():
    panel = _panel()
    table = compare_schedules(panel, _W, cost_bps=10.0)

    assert table.loc["never", "n_trades"] == 0
    assert table.loc["never", "total_cost"] == pytest.approx(0.0)
    assert table.loc["monthly", "total_cost"] > table.loc["annual", "total_cost"]


def test_comparison_shows_drift_differing_by_schedule():
    panel = _panel()
    table = compare_schedules(panel, _W)
    assert table.loc["never", "max_drift"] > table.loc["monthly", "max_drift"]


def test_schedules_produce_materially_different_outcomes():
    """If they did not, the setting would not be worth exposing."""
    panel = _panel()
    table = compare_schedules(panel, _W, cost_bps=10.0)
    spread = table["return"].max() - table["return"].min()
    assert spread > 0.001


# ---------------------------------------------------------------------------
# The commodities sleeve
# ---------------------------------------------------------------------------

def _sleeve_panel(n: int = 120, seed: int = 5) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": rng.normal(0.006, 0.046, n),
                "gold": rng.normal(0.008, 0.050, n),
                "commodities_ex_gold": rng.normal(0.001, 0.054, n),
                "cash": np.abs(rng.normal(0.0012, 0.0004, n)),
            },
            index=_dates(n),
        )
    )


_SW = {"equity": 0.6, "commodities": 0.35, "cash": 0.05}


def test_sleeve_expansion_matches_a_direct_allocation():
    """A 35% sleeve at 60/40 must be identical to 21% gold plus 14% ex-gold."""
    from core.rebalance import simulate_with_sleeve

    panel = _sleeve_panel()
    via_sleeve = simulate_with_sleeve(panel, _SW, 0.6, RebalanceSpec(Rebalance.ANNUAL, 0.0))
    direct = simulate(
        panel,
        {"equity": 0.6, "gold": 0.35 * 0.6, "commodities_ex_gold": 0.35 * 0.4, "cash": 0.05},
        RebalanceSpec(Rebalance.ANNUAL, 0.0),
    )
    assert np.allclose(
        via_sleeve.returns.to_numpy(), direct.returns.to_numpy(), atol=1e-15
    )


def test_sleeve_returns_to_the_slider_ratio_when_rebalancing():
    """With rebalancing on, the slider keeps meaning what it says."""
    from core.rebalance import simulate_with_sleeve

    panel = _sleeve_panel()
    path = simulate_with_sleeve(panel, _SW, 0.6, RebalanceSpec(Rebalance.MONTHLY, 0.0))

    sleeve_total = path.weights["gold"] + path.weights["commodities_ex_gold"]
    internal_gold_share = path.weights["gold"] / sleeve_total
    assert np.allclose(internal_gold_share.to_numpy(), 0.6, atol=1e-9)


def test_sleeve_drifts_when_rebalancing_is_off():
    """The honest consequence of choosing not to rebalance: a strong run in
    one component leaves the sleeve tilted away from the slider."""
    from core.rebalance import simulate_with_sleeve

    idx = _dates(120)
    panel = ReturnPanel(
        pd.DataFrame(
            {
                "equity": np.full(120, 0.004),
                "gold": np.full(120, 0.012),
                "commodities_ex_gold": np.full(120, 0.000),
                "cash": np.full(120, 0.001),
            },
            index=idx,
        )
    )
    path = simulate_with_sleeve(panel, _SW, 0.5, Rebalance.NEVER)

    sleeve_total = path.weights["gold"] + path.weights["commodities_ex_gold"]
    final_gold_share = float(
        (path.weights["gold"] / sleeve_total).iloc[-1]
    )
    assert final_gold_share > 0.80, "gold should have grown into the sleeve"


def test_sleeve_endpoints_hold_under_every_schedule():
    from core.rebalance import simulate_with_sleeve

    panel = _sleeve_panel()
    for schedule in Rebalance:
        path = simulate_with_sleeve(panel, _SW, 1.0, RebalanceSpec(schedule, 0.0))
        assert path.weights["commodities_ex_gold"].max() < 1e-12


def test_sleeve_simulation_rejects_a_missing_sleeve_weight():
    from core.rebalance import simulate_with_sleeve

    panel = _sleeve_panel()
    with pytest.raises(KeyError, match="commodities"):
        simulate_with_sleeve(panel, {"equity": 0.95, "cash": 0.05}, 0.5)


def test_sleeve_simulation_rejects_a_panel_without_components():
    from core.rebalance import simulate_with_sleeve

    panel = _panel()
    with pytest.raises(KeyError, match="missing sleeve components"):
        simulate_with_sleeve(panel, _SW, 0.5)
