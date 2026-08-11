"""Tests for core/portfolio.py.

The two most important tests here:

  - `test_shuffling_changes_drawdown_but_not_sharpe` demonstrates the whole
    justification for measuring the path as well as the distribution.
  - `test_risk_contributions_sum_to_total_volatility` checks Euler's theorem,
    which must hold exactly for any correct attribution.

Everything else is checked against a closed form, a hand-computed figure, or a
degenerate case with a known answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.panels import ReturnPanel
from core.portfolio import (
    analyse_drawdown,
    as_weights,
    downside_deviation,
    drawdown_series,
    portfolio_returns,
    portfolio_stats,
    risk_contributions,
)


def _dates(n: int, start: str = "2006-03-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


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
# Weight validation
# ---------------------------------------------------------------------------

def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        as_weights({"a": 0.5, "b": 0.4}, ["a", "b"])


def test_missing_asset_is_named():
    with pytest.raises(KeyError, match=r"\['b'\]"):
        as_weights({"a": 1.0}, ["a", "b"])


def test_unknown_asset_is_named():
    with pytest.raises(KeyError, match="crypto"):
        as_weights({"a": 0.5, "crypto": 0.5}, ["a"])


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="Negative weights"):
        as_weights({"a": 1.5, "b": -0.5}, ["a", "b"])


def test_weights_are_returned_in_panel_order():
    w = as_weights({"b": 0.4, "a": 0.6}, ["a", "b"])
    assert list(w.index) == ["a", "b"]


# ---------------------------------------------------------------------------
# Return and volatility
# ---------------------------------------------------------------------------

def test_two_asset_variance_matches_the_closed_form():
    """sigma^2 = w1^2 s1^2 + w2^2 s2^2 + 2 w1 w2 rho s1 s2"""
    panel = _panel().select(["equity", "bonds"])
    w1, w2 = 0.7, 0.3

    s = panel.ann_vol()
    rho = panel.corr().loc["equity", "bonds"]
    expected = np.sqrt(
        w1**2 * s["equity"] ** 2
        + w2**2 * s["bonds"] ** 2
        + 2 * w1 * w2 * rho * s["equity"] * s["bonds"]
    )

    stats = portfolio_stats(panel, {"equity": w1, "bonds": w2}, risk_free=0.0)
    assert stats.volatility == pytest.approx(expected, rel=1e-12)


def test_quadratic_form_agrees_with_the_realised_series():
    """sqrt(w' S w) and the standard deviation of the realised portfolio
    series are two independent routes to one number. They must agree."""
    panel = _panel()
    stats = portfolio_stats(panel, _W)

    ann_cov = panel.ann_cov().to_numpy()
    wv = stats.weights.to_numpy()
    from_matrix = np.sqrt(wv @ ann_cov @ wv)

    assert stats.volatility == pytest.approx(from_matrix, rel=1e-12)


def test_single_asset_portfolio_reduces_to_that_asset():
    panel = _panel()
    stats = portfolio_stats(
        panel, {"equity": 1.0, "bonds": 0.0, "gold": 0.0, "cash": 0.0}, risk_free=0.0
    )
    assert stats.volatility == pytest.approx(panel.ann_vol()["equity"], rel=1e-12)
    assert stats.realised_return == pytest.approx(
        panel.ann_return()["equity"], rel=1e-12
    )


def test_diversification_reduces_volatility():
    """A mix must be calmer than the weighted average of its parts, unless
    everything correlates perfectly."""
    panel = _panel()
    stats = portfolio_stats(panel, _W)
    naive = float(stats.weights @ panel.ann_vol())
    assert stats.volatility < naive


def test_portfolio_returns_are_the_weighted_sum():
    panel = _panel()
    manual = panel.returns.mul(pd.Series(_W), axis=1).sum(axis=1)
    assert np.allclose(
        portfolio_returns(panel, _W).to_numpy(), manual.to_numpy(), atol=1e-15
    )


def test_model_and_realised_return_both_reported():
    panel = _panel()
    stats = portfolio_stats(panel, _W)
    assert stats.return_gap == pytest.approx(
        stats.realised_return - stats.model_return
    )


# ---------------------------------------------------------------------------
# Sharpe and Sortino
# ---------------------------------------------------------------------------

def test_sharpe_matches_its_definition():
    panel = _panel()
    stats = portfolio_stats(panel, _W)
    assert stats.sharpe == pytest.approx(stats.excess_return / stats.volatility)


def test_sortino_matches_its_definition():
    panel = _panel()
    stats = portfolio_stats(panel, _W)
    assert stats.sortino == pytest.approx(
        stats.excess_return / stats.downside_deviation
    )


def test_sortino_exceeds_sharpe_for_upside_skewed_returns():
    """Sortino ignores upside movement, so a portfolio that jumps upward is
    not punished for it. Its denominator must be the smaller one.

    The fixture includes real down months -- without any, downside deviation
    is exactly zero and the comparison is degenerate rather than informative.
    """
    idx = _dates(60)
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0.004, 0.015, 60), index=idx)
    rets.iloc[::10] = 0.15  # occasional large gains, no matching large losses
    panel = ReturnPanel(pd.DataFrame({"a": rets, "cash": np.full(60, 0.001)}, index=idx))

    stats = portfolio_stats(panel, {"a": 1.0, "cash": 0.0})
    assert stats.downside_deviation > 0, "fixture must contain shortfalls"
    assert stats.downside_deviation < stats.volatility
    assert stats.sortino > stats.sharpe


def test_sortino_is_infinite_when_nothing_falls_short():
    """A portfolio that never underperformed cash has an unbounded Sortino.
    Reporting zero would say the opposite, and an optimizer maximising the
    ratio would avoid exactly the portfolio it should prefer."""
    idx = _dates(36)
    panel = ReturnPanel(
        pd.DataFrame(
            {"a": np.full(36, 0.010), "cash": np.full(36, 0.001)}, index=idx
        )
    )
    stats = portfolio_stats(panel, {"a": 1.0, "cash": 0.0})

    assert stats.downside_deviation == pytest.approx(0.0)
    assert stats.sortino == float("inf")


def test_sharpe_is_negative_infinite_when_underperforming_with_no_movement():
    """A flat series below cash has zero volatility and a negative excess
    return. Reporting 0.0 would rank it alongside a portfolio that exactly
    matched cash, which is a materially different outcome.

    Note this branch is reachable for Sharpe but not for Sortino: a negative
    excess return implies at least one month fell short of the target, which
    guarantees a positive downside deviation.
    """
    idx = _dates(36)
    panel = ReturnPanel(
        pd.DataFrame(
            {"a": np.full(36, 0.0001), "cash": np.full(36, 0.002)}, index=idx
        )
    )
    stats = portfolio_stats(panel, {"a": 1.0, "cash": 0.0})

    assert stats.volatility == pytest.approx(0.0)
    assert stats.excess_return < 0
    assert stats.sharpe == float("-inf")
    assert stats.sortino < 0 and np.isfinite(stats.sortino)


def test_downside_deviation_is_zero_when_nothing_falls_short():
    idx = _dates(24)
    rets = pd.Series(np.full(24, 0.01), index=idx)
    assert downside_deviation(rets, 0.001, 12) == pytest.approx(0.0)


def test_downside_deviation_hand_computed():
    """Two months at -4% against a zero target, two at +4%. Mean squared
    shortfall is (0.04^2 + 0.04^2) / 4, annualised by sqrt(12)."""
    idx = _dates(4)
    rets = pd.Series([0.04, -0.04, 0.04, -0.04], index=idx)
    expected = np.sqrt((2 * 0.04**2) / 4) * np.sqrt(12)
    assert downside_deviation(rets, 0.0, 12) == pytest.approx(expected)


def test_downside_deviation_averages_over_all_periods():
    """Dividing by the count of bad months instead of all months would make
    rare-but-severe losses look better than frequent mild ones."""
    idx = _dates(12)
    rare_severe = pd.Series([0.0] * 11 + [-0.30], index=idx)
    frequent_mild = pd.Series([-0.02] * 12, index=idx)

    assert downside_deviation(rare_severe, 0.0, 12) > downside_deviation(
        frequent_mild, 0.0, 12
    )


def test_time_varying_risk_free_is_used():
    """A fixed early-sample rate would credit a cash-heavy portfolio with an
    excess return it never earned once rates rose."""
    idx = _dates(120)
    cash = pd.Series(
        np.concatenate([np.full(60, 0.0001), np.full(60, 0.0041)]), index=idx
    )
    panel = ReturnPanel(
        pd.DataFrame({"a": np.full(120, 0.004), "cash": cash}, index=idx)
    )

    all_cash = portfolio_stats(panel, {"a": 0.0, "cash": 1.0})
    assert abs(all_cash.excess_return) < 0.002
    assert abs(all_cash.sharpe) < 0.05


def test_flat_portfolio_matching_cash_scores_zero():
    """No excess return and no movement is genuinely zero, not infinite."""
    idx = _dates(36)
    panel = ReturnPanel(
        pd.DataFrame(
            {"a": np.full(36, 0.001), "cash": np.full(36, 0.001)}, index=idx
        )
    )
    stats = portfolio_stats(panel, {"a": 1.0, "cash": 0.0})
    assert stats.sharpe == 0.0
    assert stats.sortino == 0.0


# ---------------------------------------------------------------------------
# Drawdown -- the path statistics
# ---------------------------------------------------------------------------

def test_drawdown_hand_computed():
    """Up 20%, down 40%, up 30%. Peak 1.20, trough 0.72, so -40% exactly.
    Recovery needs 1.20; 0.72 * 1.30 = 0.936, so it does not recover."""
    idx = _dates(4)
    rets = pd.Series([0.0, 0.20, -0.40, 0.30], index=idx)
    dd = analyse_drawdown(rets)

    assert dd.max_drawdown == pytest.approx(-0.40)
    assert dd.peak_date == idx[1]
    assert dd.trough_date == idx[2]
    assert dd.months_to_trough == 1
    assert not dd.recovered
    assert dd.months_to_recover is None


def test_drawdown_recovery_is_detected():
    """Same shape, but the final gain is enough to regain the prior peak."""
    idx = _dates(4)
    rets = pd.Series([0.0, 0.20, -0.40, 0.70], index=idx)
    dd = analyse_drawdown(rets)

    assert dd.max_drawdown == pytest.approx(-0.40)
    assert dd.recovered
    assert dd.recovery_date == idx[3]
    assert dd.months_to_recover == 1


def test_recovery_requires_the_prior_peak_not_just_a_rise():
    """Rising off the bottom is not recovery. The portfolio is whole only when
    the loss is undone."""
    idx = _dates(5)
    rets = pd.Series([0.0, 0.20, -0.40, 0.10, 0.10], index=idx)
    dd = analyse_drawdown(rets)
    assert not dd.recovered


def test_monotonic_rise_has_no_drawdown():
    idx = _dates(24)
    dd = analyse_drawdown(pd.Series(np.full(24, 0.01), index=idx))
    assert dd.max_drawdown == pytest.approx(0.0)
    assert dd.months_underwater == 0


def test_drawdown_series_is_never_positive():
    panel = _panel()
    series = drawdown_series(portfolio_returns(panel, _W))
    assert series.max() <= 1e-12
    assert series.min() == pytest.approx(
        analyse_drawdown(portfolio_returns(panel, _W)).max_drawdown
    )


def test_months_underwater_counts_the_whole_period():
    """Two separate episodes below a high-water mark both count."""
    idx = _dates(8)
    rets = pd.Series([0.10, -0.05, 0.10, 0.0, -0.05, -0.05, 0.20, 0.0], index=idx)
    dd = analyse_drawdown(rets)
    assert dd.months_underwater >= 3


def test_shuffling_changes_drawdown_but_not_sharpe():
    """The justification for the entire path-statistics section.

    Sharpe and Sortino are invariant to the order months occur in. Drawdown is
    not. Twelve scattered bad months and twelve consecutive ones are the same
    distribution and a completely different experience.
    """
    rng = np.random.default_rng(5)
    idx = _dates(120)
    values = rng.normal(0.006, 0.05, 120)

    ordered = pd.Series(np.sort(values)[::-1], index=idx)  # all gains, then all losses
    shuffled = pd.Series(rng.permutation(values), index=idx)

    panel_a = ReturnPanel(
        pd.DataFrame({"a": ordered, "cash": np.full(120, 0.001)}, index=idx)
    )
    panel_b = ReturnPanel(
        pd.DataFrame({"a": shuffled, "cash": np.full(120, 0.001)}, index=idx)
    )

    stats_a = portfolio_stats(panel_a, {"a": 1.0, "cash": 0.0})
    stats_b = portfolio_stats(panel_b, {"a": 1.0, "cash": 0.0})

    assert stats_a.volatility == pytest.approx(stats_b.volatility, rel=1e-12)
    assert stats_a.max_drawdown < stats_b.max_drawdown - 0.05


def test_drawdown_needs_two_observations():
    with pytest.raises(ValueError, match="at least 2"):
        analyse_drawdown(pd.Series([0.01], index=_dates(1)))


# ---------------------------------------------------------------------------
# Risk contributions
# ---------------------------------------------------------------------------

def test_risk_contributions_sum_to_total_volatility():
    """Euler's theorem. Volatility is homogeneous of degree one in the
    weights, so the weighted marginal contributions add back exactly."""
    panel = _panel()
    rc = risk_contributions(panel, _W)
    stats = portfolio_stats(panel, _W)

    assert rc["risk_contribution"].sum() == pytest.approx(stats.volatility, rel=1e-12)
    assert rc["pct_of_risk"].sum() == pytest.approx(1.0, rel=1e-12)


def test_weight_is_not_risk():
    """The point of the table. A volatile, correlated asset carries more risk
    than its weight suggests; a calm one carries less."""
    panel = _panel()
    rc = risk_contributions(panel, _W)

    assert rc.loc["equity", "pct_of_risk"] > rc.loc["equity", "weight"]
    assert rc.loc["cash", "pct_of_risk"] < rc.loc["cash", "weight"]


def test_correlated_asset_carries_more_risk_than_its_weight():
    """Two assets that move together do not diversify each other, so both
    carry more risk than an independent asset of the same volatility would."""
    idx = _dates(240)
    rng = np.random.default_rng(9)
    base = rng.normal(0.006, 0.045, 240)
    panel = ReturnPanel(
        pd.DataFrame(
            {
                "a": base,
                "twin": base + rng.normal(0, 0.004, 240),  # ~0.99 correlated
                "independent": rng.normal(0.006, 0.045, 240),
            },
            index=idx,
        )
    )
    rc = risk_contributions(panel, {"a": 1 / 3, "twin": 1 / 3, "independent": 1 / 3})
    assert rc.loc["a", "pct_of_risk"] > rc.loc["independent", "pct_of_risk"]


def test_single_asset_carries_all_the_risk():
    panel = _panel()
    rc = risk_contributions(
        panel, {"equity": 1.0, "bonds": 0.0, "gold": 0.0, "cash": 0.0}
    )
    assert rc.loc["equity", "pct_of_risk"] == pytest.approx(1.0, rel=1e-12)


def test_zero_weight_carries_no_risk():
    panel = _panel()
    rc = risk_contributions(panel, {**_W, "gold": 0.0, "equity": 0.7})
    assert rc.loc["gold", "risk_contribution"] == pytest.approx(0.0)


def test_risk_contributions_reject_bad_weights():
    panel = _panel()
    with pytest.raises(ValueError, match="sum to 1.0"):
        risk_contributions(panel, {"equity": 0.5, "bonds": 0.2, "gold": 0.2, "cash": 0.2})


def test_drawdown_from_the_very_first_month():
    """A portfolio that falls immediately must measure its decline from
    starting capital, not from the first month's closing value.

    Without the implicit starting point, the running maximum begins already
    depressed and the reported drawdown is understated. This is the case that
    matters most: a user selecting a period beginning at a market peak.
    """
    idx = _dates(3)
    rets = pd.Series([-0.10, -0.10, 0.0], index=idx)
    dd = analyse_drawdown(rets)

    assert dd.max_drawdown == pytest.approx(-0.19)
    assert dd.trough_date == idx[1]
    assert dd.peak_date < idx[0], "the peak is the starting capital"


def test_immediate_decline_recovery_is_measured_from_starting_capital():
    idx = _dates(4)
    rets = pd.Series([-0.20, 0.0, 0.10, 0.15], index=idx)
    dd = analyse_drawdown(rets)

    assert dd.max_drawdown == pytest.approx(-0.20)
    # 0.8 * 1.10 * 1.15 = 1.012, so it regains 1.0 in the final month.
    assert dd.recovered
    assert dd.recovery_date == idx[3]


def test_drawdown_series_covers_every_return_date():
    """The implicit origin seeds the running maximum but is not reported --
    no decline has occurred before the first return."""
    panel = _panel()
    rets = portfolio_returns(panel, _W)
    series = drawdown_series(rets)

    assert series.index.equals(rets.index)
    assert series.iloc[0] <= 1e-12


def test_first_month_gain_leaves_no_drawdown():
    """The mirror case: rising from the outset means the starting point is
    never a peak that gets breached."""
    idx = _dates(3)
    dd = analyse_drawdown(pd.Series([0.05, 0.05, 0.05], index=idx))
    assert dd.max_drawdown == pytest.approx(0.0)
    assert dd.months_underwater == 0


def test_the_two_return_measures_are_distinct_and_both_available():
    """`realised_return` is geometric, `excess_return` is an annualised
    arithmetic mean. They answer different questions and neither should be
    silently substituted for the other."""
    panel = _panel()
    stats = portfolio_stats(panel, _W)

    manual_arithmetic = float(
        (stats.returns - panel.returns["cash"]).mean() * panel.periods_per_year
    )
    manual_geometric = float(
        (1 + stats.returns).prod() ** (panel.periods_per_year / len(stats.returns)) - 1
    )

    assert stats.excess_return == pytest.approx(manual_arithmetic, rel=1e-12)
    assert stats.realised_return == pytest.approx(manual_geometric, rel=1e-12)


def test_scalar_risk_free_is_deannualised_not_applied_monthly():
    """A 12% annual rate must not be subtracted as 12% from every month."""
    idx = _dates(60)
    panel = ReturnPanel(
        pd.DataFrame({"a": np.full(60, 0.01), "cash": np.full(60, 0.0)}, index=idx)
    )
    stats = portfolio_stats(panel, {"a": 1.0, "cash": 0.0}, risk_free=0.12)

    monthly_rf = 1.12 ** (1 / 12) - 1
    assert stats.excess_return == pytest.approx((0.01 - monthly_rf) * 12, rel=1e-9)


def test_risk_free_series_must_cover_the_period():
    panel = _panel()
    short = pd.Series([0.001] * 10, index=panel.returns.index[:10])
    with pytest.raises(ValueError, match="does not cover"):
        portfolio_stats(panel, _W, risk_free=short)
