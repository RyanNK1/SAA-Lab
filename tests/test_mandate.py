"""Tests for core/mandate.py.

Two anchors:

  - `test_every_qualifying_allocation_actually_meets_the_mandate` -- if a
    returned allocation breaks a limit, the whole layer is worthless, because
    the point of a mandate is that it is binding.
  - `test_an_unreachable_target_is_diagnosed_not_just_refused` -- "impossible"
    is a true answer and a useless one. The value is in saying what would have
    to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.constraints import Constraints, GroupLimit
from core.mandate import (
    RANKABLE,
    Mandate,
    frontier_of_mandates,
    solve_mandate,
)
from core.panels import ReturnPanel

FAST = 3_000


def _panel(n: int = 240, seed: int = 11) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": rng.normal(0.0065, 0.046, n),
                "fixed_income": rng.normal(0.0025, 0.013, n),
                "private_equity": rng.normal(0.0070, 0.058, n),
                "commodities": rng.normal(0.0040, 0.042, n),
                "cash": np.abs(rng.normal(0.0012, 0.0004, n)),
            },
            index=pd.date_range("2006-03-31", periods=n, freq="ME"),
        )
    )


# ---------------------------------------------------------------------------
# Mandate validation
# ---------------------------------------------------------------------------

def test_a_mandate_needs_at_least_one_requirement():
    with pytest.raises(ValueError, match="at least one requirement"):
        Mandate()


def test_percentages_are_rejected_in_place_of_fractions():
    """6 instead of 0.06 is the obvious slip, and it would silently produce a
    mandate nothing can meet."""
    with pytest.raises(ValueError, match="fraction"):
        Mandate(target_return=6.0)
    with pytest.raises(ValueError, match="fraction"):
        Mandate(max_volatility=10.0)


def test_a_positive_drawdown_limit_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        Mandate(max_drawdown=0.20)


def test_describe_lists_every_requirement():
    mandate = Mandate(
        target_return=0.06,
        max_volatility=0.10,
        constraints=Constraints(floors={"cash": 0.05}),
    )
    text = mandate.describe()
    assert "return >= 6.0%" in text
    assert "volatility <= 10.0%" in text
    assert "cash >= 5%" in text


# ---------------------------------------------------------------------------
# Feasible mandates
# ---------------------------------------------------------------------------

def test_every_qualifying_allocation_actually_meets_the_mandate():
    """The anchor. A mandate that returns allocations breaking its own limits
    is worse than no mandate at all."""
    panel = _panel()
    mandate = Mandate(
        target_return=0.04,
        max_volatility=0.10,
        max_drawdown=-0.25,
        constraints=Constraints(floors={"cash": 0.02}, caps={"private_equity": 0.25}),
    )
    result = solve_mandate(panel, mandate, n_samples=FAST)

    assert result.feasible
    assert (result.qualifying["realised_return"] >= 0.04 - 1e-9).all()
    assert (result.qualifying["volatility"] <= 0.10 + 1e-9).all()
    assert (result.qualifying["max_drawdown"] >= -0.25 - 1e-9).all()

    for _, row in result.qualifying.iterrows():
        weights = row[panel.assets]
        assert mandate.constraints.satisfied_by(weights)
        assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_a_lenient_mandate_admits_many_allocations():
    """The usual case: a mandate is satisfied by hundreds of allocations, not
    one. Presenting a single answer would be a fiction."""
    panel = _panel()
    result = solve_mandate(
        panel, Mandate(max_volatility=0.30), n_samples=FAST
    )
    assert result.n_qualifying > 100


def test_tightening_a_requirement_never_admits_more():
    panel = _panel()
    loose = solve_mandate(panel, Mandate(max_volatility=0.15), n_samples=FAST)
    tight = solve_mandate(panel, Mandate(max_volatility=0.08), n_samples=FAST)
    assert tight.n_qualifying <= loose.n_qualifying


def test_each_requirement_filters_independently():
    panel = _panel()
    just_return = solve_mandate(
        panel, Mandate(target_return=0.05), n_samples=FAST
    )
    both = solve_mandate(
        panel, Mandate(target_return=0.05, max_volatility=0.08), n_samples=FAST
    )
    assert both.n_qualifying <= just_return.n_qualifying


# ---------------------------------------------------------------------------
# Ranking is the user's choice
# ---------------------------------------------------------------------------

def test_ranking_orders_correctly_in_both_directions():
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)

    by_return = result.ranked("realised_return")
    assert by_return["realised_return"].is_monotonic_decreasing

    by_vol = result.ranked("volatility")
    assert by_vol["volatility"].is_monotonic_increasing


def test_drawdown_ranking_puts_the_shallowest_first():
    """Drawdowns are negative, so closer to zero is better."""
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)
    ordered = result.ranked("max_drawdown")
    assert ordered["max_drawdown"].is_monotonic_decreasing
    assert ordered["max_drawdown"].iloc[0] >= ordered["max_drawdown"].iloc[-1]


def test_different_rankings_give_different_winners():
    """If they agreed, letting the user choose would be pointless."""
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)

    top_return = result.ranked("realised_return").iloc[0][panel.assets]
    top_calm = result.ranked("volatility").iloc[0][panel.assets]
    assert not np.allclose(top_return.to_numpy(), top_calm.to_numpy(), atol=1e-3)


def test_ranking_by_an_unknown_measure_is_rejected():
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)
    with pytest.raises(KeyError, match="Cannot rank by"):
        result.ranked("vibes")


def test_every_rankable_column_is_present_in_the_output():
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)
    for column in RANKABLE:
        assert column in result.qualifying.columns


def test_limit_caps_the_number_returned():
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)
    assert len(result.ranked("sharpe", limit=10)) == 10


# ---------------------------------------------------------------------------
# The envelope and headroom
# ---------------------------------------------------------------------------

def test_the_envelope_brackets_every_qualifying_allocation():
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.15), n_samples=FAST)
    envelope = result.envelope()

    for asset in panel.assets:
        assert envelope.loc[asset, "min"] <= envelope.loc[asset, "median"]
        assert envelope.loc[asset, "median"] <= envelope.loc[asset, "max"]
        assert result.qualifying[asset].min() == pytest.approx(
            envelope.loc[asset, "min"]
        )


def test_headroom_is_non_negative_for_every_qualifying_allocation():
    """Anything qualifying must have slack, or zero slack, against each limit
    -- never a shortfall."""
    panel = _panel()
    mandate = Mandate(target_return=0.04, max_volatility=0.12, max_drawdown=-0.30)
    result = solve_mandate(panel, mandate, n_samples=FAST)
    headroom = result.headroom()

    assert (headroom >= -1e-9).all().all()


# ---------------------------------------------------------------------------
# Infeasibility, and what to do about it
# ---------------------------------------------------------------------------

def test_an_unreachable_target_is_diagnosed_not_just_refused():
    """The second anchor. 'Impossible' is true and useless; the value is in
    saying how far the target would have to fall."""
    panel = _panel()
    mandate = Mandate(target_return=0.40, max_volatility=0.10)
    result = solve_mandate(panel, mandate, n_samples=FAST)

    assert not result.feasible
    assert result.relaxations

    reported = {r.what for r in result.relaxations}
    assert "target return" in reported

    target_fix = next(r for r in result.relaxations if r.what == "target return")
    assert target_fix.required < mandate.target_return


def test_an_impossible_budget_suggests_raising_it():
    panel = _panel()
    mandate = Mandate(target_return=0.06, max_volatility=0.005)
    result = solve_mandate(panel, mandate, n_samples=FAST)

    assert not result.feasible
    budget_fix = [r for r in result.relaxations if r.what == "volatility budget"]
    assert budget_fix
    assert budget_fix[0].required > mandate.max_volatility


def test_a_blocking_constraint_is_identified():
    """When a policy rule is what makes the mandate impossible, say which."""
    panel = _panel()
    mandate = Mandate(
        target_return=0.055,
        constraints=Constraints(floors={"cash": 0.95}),
    )
    result = solve_mandate(panel, mandate, n_samples=FAST)

    assert not result.feasible
    assert any("cash" in r.what for r in result.relaxations)


def test_the_explanation_is_readable():
    panel = _panel()
    result = solve_mandate(
        panel, Mandate(target_return=0.40, max_volatility=0.10), n_samples=FAST
    )
    text = result.explain()
    assert "No allocation meets the mandate" in text
    assert "->" in text


def test_a_feasible_mandate_explains_how_many_qualified():
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)
    assert "meet the mandate" in result.explain()


def test_jointly_impossible_policy_limits_fail_loudly():
    panel = _panel()
    mandate = Mandate(
        target_return=0.05,
        constraints=Constraints(
            floors={"cash": 0.60},
            groups=(
                GroupLimit("risky", ("equity", "private_equity"), minimum=0.60),
            ),
        ),
    )
    with pytest.raises((RuntimeError, ValueError)):
        solve_mandate(panel, mandate, n_samples=500)


# ---------------------------------------------------------------------------
# Sweeping the target
# ---------------------------------------------------------------------------

def test_the_frontier_finds_where_a_budget_stops_working():
    """More useful than testing one target: it answers how much could have
    been asked for, not only whether this could."""
    panel = _panel()
    frontier = frontier_of_mandates(
        panel, targets=[0.02, 0.04, 0.06, 0.20], max_volatility=0.10, n_samples=FAST
    )

    assert frontier["feasible"].iloc[0]
    assert not frontier["feasible"].iloc[-1]


def test_feasibility_is_monotonic_in_the_target():
    """Once a target becomes unreachable, higher ones stay unreachable."""
    panel = _panel()
    frontier = frontier_of_mandates(
        panel,
        targets=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.12, 0.20],
        max_volatility=0.12,
        n_samples=FAST,
    )
    feasible = frontier["feasible"].to_numpy()
    first_failure = np.argmax(~feasible) if (~feasible).any() else len(feasible)
    assert not feasible[first_failure:].any()


def test_the_frontier_reports_the_best_drawdown_where_feasible():
    panel = _panel()
    frontier = frontier_of_mandates(
        panel, targets=[0.03, 0.05], max_volatility=0.12, n_samples=FAST
    )
    workable = frontier[frontier["feasible"]]
    assert (workable["best_drawdown"] <= 0).all()


def test_diagnosis_works_when_only_one_requirement_is_set():
    """The diagnosis asks what would be reachable with each limit lifted. With
    only one requirement, lifting it would leave a mandate with nothing to
    solve -- so the question must be asked by filtering, not by constructing
    an empty mandate."""
    panel = _panel()
    for mandate in (
        Mandate(target_return=0.40),
        Mandate(max_volatility=0.0005),
    ):
        result = solve_mandate(panel, mandate, n_samples=FAST)
        assert not result.feasible
        assert isinstance(result.explain(), str)


def test_a_drawdown_limit_alone_is_always_satisfiable():
    """Cash has essentially no drawdown, so a drawdown limit on its own can
    always be met -- by holding cash and earning nothing. It only becomes
    binding alongside a return target, which is the realistic mandate."""
    panel = _panel()
    alone = solve_mandate(panel, Mandate(max_drawdown=-0.001), n_samples=FAST)
    assert alone.feasible
    assert alone.ranked("realised_return").iloc[0]["cash"] > 0.8

    with_target = solve_mandate(
        panel, Mandate(max_drawdown=-0.001, target_return=0.06), n_samples=FAST
    )
    assert not with_target.feasible
    assert with_target.relaxations


def test_ignoring_a_requirement_loosens_the_mask():
    """Directly: the mask must admit at least as many allocations when one
    limit is set aside."""
    panel = _panel()
    from core.optimize import _measure_samples, sample_allocations

    samples = sample_allocations(len(panel.assets), 500, seed=1)
    measured = _measure_samples(panel, samples, None)

    mandate = Mandate(target_return=0.05, max_volatility=0.08)
    strict = mandate.qualifies(measured).sum()
    relaxed = mandate.qualifies(measured, ignoring="max_volatility").sum()
    assert relaxed >= strict


# ---------------------------------------------------------------------------
# Path measures
# ---------------------------------------------------------------------------

def test_recovery_and_underwater_are_rankable():
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)

    assert "months_to_recover" in result.qualifying.columns
    assert "months_underwater" in result.qualifying.columns

    by_recovery = result.ranked("months_to_recover")
    assert by_recovery["months_to_recover"].is_monotonic_increasing


def test_measured_rows_stay_numeric():
    """Every column is a float, so pulling weights out of a result behaves
    like numbers. A single boolean column alongside them would make each
    extracted row object-dtype and quietly break arithmetic on it."""
    panel = _panel()
    result = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=1_000)
    row = result.ranked("sharpe").iloc[0]
    weights = row[panel.assets].to_numpy(dtype=float)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_never_recovered_is_infinity_not_a_plausible_number():
    """Encoding 'never recovered' as the periods remaining after the trough
    would be a lower bound -- and would rank an allocation that never got back
    to its peak ahead of one that genuinely recovered more slowly."""
    from core.optimize import _measure_samples, sample_allocations
    from core.portfolio import portfolio_stats

    panel = _panel()
    samples = sample_allocations(len(panel.assets), 60, seed=1)
    measured = _measure_samples(panel, samples, None)

    for i in range(len(samples)):
        direct = portfolio_stats(panel, pd.Series(samples[i], index=panel.assets))
        sampled = measured.loc[i, "months_to_recover"]

        if direct.drawdown.recovered:
            assert sampled == direct.drawdown.months_to_recover
        else:
            assert np.isinf(sampled)

        assert measured.loc[i, "months_underwater"] == direct.drawdown.months_underwater


def test_a_recovery_limit_filters_slow_recoverers():
    panel = _panel()
    unlimited = solve_mandate(panel, Mandate(max_volatility=0.20), n_samples=FAST)
    limited = solve_mandate(
        panel, Mandate(max_volatility=0.20, max_recovery_months=12), n_samples=FAST
    )

    assert limited.n_qualifying < unlimited.n_qualifying
    assert (limited.qualifying["months_to_recover"] <= 12).all()


def test_a_recovery_limit_below_one_month_is_rejected():
    with pytest.raises(ValueError, match="max_recovery_months"):
        Mandate(max_recovery_months=0)


# ---------------------------------------------------------------------------
# Rebalancing is part of the mandate
# ---------------------------------------------------------------------------

def test_the_rebalancing_schedule_changes_which_allocations_qualify():
    """An instruction that does not say how the portfolio is held is
    incomplete: 'at least 4% a year' means something different for a portfolio
    corrected annually than for one left to drift."""
    from core.config import Rebalance
    from core.rebalance import RebalanceSpec

    panel = _panel()
    counts = {}
    for schedule in (Rebalance.MONTHLY, Rebalance.ANNUAL, Rebalance.NEVER):
        cost = 0.0 if schedule is Rebalance.MONTHLY else 10.0
        mandate = Mandate(
            target_return=0.04,
            max_volatility=0.10,
            rebalance=RebalanceSpec(schedule, cost),
        )
        # Non-monthly schedules simulate each allocation individually rather
        # than in one matrix product, so the budget here is deliberately small.
        counts[schedule] = solve_mandate(panel, mandate, n_samples=200).n_qualifying

    assert len(set(counts.values())) > 1, f"schedules gave identical results: {counts}"


def test_the_default_schedule_uses_the_fast_measurement_path():
    """The default must be the one the vectorised measurement can handle. Any
    other schedule simulates each allocation in a loop, which is orders of
    magnitude slower -- acceptable as a deliberate choice, not as a default
    nobody noticed."""
    from core.config import Rebalance

    mandate = Mandate(target_return=0.05)
    assert mandate.rebalance.schedule is Rebalance.MONTHLY
    assert mandate.rebalance.cost_bps == 0.0


def test_the_schedule_appears_in_the_description():
    from core.config import Rebalance
    from core.rebalance import RebalanceSpec

    mandate = Mandate(
        target_return=0.06, rebalance=RebalanceSpec(Rebalance.ANNUAL, 10.0)
    )
    assert "annual rebalancing" in mandate.describe()


# ---------------------------------------------------------------------------
# Across periods
# ---------------------------------------------------------------------------

def _periods(panel: ReturnPanel):
    from core.periods import Period

    index = panel.returns.index
    return [
        Period("early", index[0], index[80]),
        Period("middle", index[80], index[160]),
        Period("late", index[160], index[-1]),
    ]


def test_surviving_every_period_is_harder_than_surviving_one():
    """The point of the cross-period test: meeting a mandate once is a
    hindsight result."""
    from core.mandate import solve_mandate_across_periods

    panel = _panel()
    mandate = Mandate(target_return=0.04, max_volatility=0.12)
    result = solve_mandate_across_periods(panel, mandate, _periods(panel), n_samples=FAST)

    counts = result.survival_counts()
    assert len(result.survivors) <= counts.min()


def test_survivors_meet_the_mandate_in_every_period():
    from core.mandate import solve_mandate_across_periods
    from core.portfolio import portfolio_stats

    panel = _panel()
    periods = _periods(panel)
    mandate = Mandate(target_return=0.03, max_volatility=0.15)
    result = solve_mandate_across_periods(panel, mandate, periods, n_samples=FAST)

    if not result.any_survivors:
        pytest.skip("nothing survived; the check needs at least one survivor")

    weights = result.survivors.iloc[0]
    for period in periods:
        window = panel.between(period.start, period.end)
        cash = window.returns["cash"] if "cash" in window.assets else None
        stats = portfolio_stats(window, weights, risk_free=cash)
        assert stats.realised_return >= 0.03 - 1e-9
        assert stats.volatility <= 0.15 + 1e-9


def test_the_binding_period_is_identified():
    from core.mandate import solve_mandate_across_periods

    panel = _panel()
    result = solve_mandate_across_periods(
        panel, Mandate(target_return=0.04, max_volatility=0.12), _periods(panel),
        n_samples=FAST,
    )
    assert "binding period" in result.explain()


def test_an_impossible_mandate_survives_nowhere():
    from core.mandate import solve_mandate_across_periods

    panel = _panel()
    result = solve_mandate_across_periods(
        panel, Mandate(target_return=0.50), _periods(panel), n_samples=1_000
    )
    assert not result.any_survivors
    assert "No allocation met the mandate in every period" in result.explain()
