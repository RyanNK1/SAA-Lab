"""Tests for core/optimize.py.

The two anchors:

  - `test_min_variance_matches_the_closed_form` checks the exact solver
    against an algebraic solution, which is a genuine external reference
    rather than the code agreeing with itself.
  - `test_no_random_allocation_beats_the_optimizer` checks every objective
    against brute force, which is what catches a solver quietly stopping early.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Objective
from core.optimize import (
    Method,
    efficient_frontier,
    optimize,
    optimize_all,
    sample_allocations,
    score,
)
from core.panels import ReturnPanel
from core.portfolio import portfolio_stats

FAST = 2_000  # keep the suite quick; correctness does not need the full budget


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


def _closed_form_min_variance(panel: ReturnPanel) -> np.ndarray:
    cov = panel.ann_cov().to_numpy()
    inverse = np.linalg.inv(cov)
    ones = np.ones(len(cov))
    w = inverse @ ones
    return w / (ones @ w)


# ---------------------------------------------------------------------------
# Exact solvers
# ---------------------------------------------------------------------------

def test_min_variance_matches_the_closed_form():
    """The analytic solution is the ground truth. Valid only where the
    unconstrained answer happens to be long-only, which is checked first."""
    panel = _panel()
    analytic = _closed_form_min_variance(panel)
    assert (analytic > 0).all(), "test only meaningful when the analytic answer is long-only"

    result = optimize(panel, Objective.MIN_VOLATILITY, n_samples=FAST)
    assert result.method is Method.EXACT
    assert result.weights.to_numpy() == pytest.approx(analytic, abs=1e-6)


def test_smooth_objectives_report_themselves_as_exact():
    panel = _panel()
    for objective in (Objective.MIN_VOLATILITY, Objective.MAX_SHARPE):
        assert optimize(panel, objective, n_samples=FAST).method is Method.EXACT


def test_path_objectives_report_themselves_as_sampled():
    """An answer from a search is a strong candidate, not a proven optimum,
    and the result must say so."""
    panel = _panel()
    for objective in (Objective.MAX_SORTINO, Objective.MIN_DRAWDOWN):
        assert optimize(panel, objective, n_samples=FAST).method is Method.SAMPLED


# ---------------------------------------------------------------------------
# Against brute force
# ---------------------------------------------------------------------------

def test_no_random_allocation_beats_the_optimizer():
    """The check that catches a solver stopping early. For every objective,
    2,000 random allocations must all be worse than the optimizer's answer."""
    panel = _panel()
    rng = np.random.default_rng(99)
    draws = rng.dirichlet(np.ones(len(panel.assets)), size=2_000)

    for objective in Objective:
        result = optimize(panel, objective, n_samples=FAST)
        best = result.value

        for draw in draws:
            stats = portfolio_stats(panel, pd.Series(draw, index=panel.assets))
            value = score(stats, objective)
            if not np.isfinite(value):
                continue

            if objective is Objective.MIN_VOLATILITY:
                assert value >= best - 1e-6
            elif objective is Objective.MIN_DRAWDOWN:
                assert value <= best + 1e-3
            else:
                assert value <= best + 1e-3


def test_min_variance_is_not_beaten_by_the_sample_set():
    panel = _panel()
    result = optimize(panel, Objective.MIN_VOLATILITY, n_samples=FAST)
    assert result.stats.volatility <= result.near_optimal["volatility"].min() + 1e-9


# ---------------------------------------------------------------------------
# The objectives differ
# ---------------------------------------------------------------------------

def test_each_objective_returns_a_different_allocation():
    """If two objectives agree exactly, one of them is not doing its job."""
    panel = _panel()
    results = optimize_all(panel, n_samples=FAST)

    weights = {o: r.weights.round(4).to_numpy() for o, r in results.items()}
    pairs = [
        (Objective.MIN_VOLATILITY, Objective.MAX_SHARPE),
        (Objective.MAX_SHARPE, Objective.MIN_DRAWDOWN),
    ]
    for a, b in pairs:
        assert not np.allclose(weights[a], weights[b], atol=1e-3), f"{a} == {b}"


def test_min_volatility_prefers_the_calmest_asset():
    panel = _panel()
    weights = optimize(panel, Objective.MIN_VOLATILITY, n_samples=FAST).weights
    assert weights["cash"] > weights["equity"]


def test_min_drawdown_hides_in_cash():
    """A correct answer to the question asked and a useless one to the question
    meant -- which is why weight caps exist. Documented as a test so the
    behaviour is not mistaken for a bug later."""
    panel = _panel()
    weights = optimize(panel, Objective.MIN_DRAWDOWN, n_samples=FAST).weights
    assert weights["cash"] > 0.7


def test_capping_forces_diversification():
    panel = _panel()
    weights = optimize(
        panel, Objective.MIN_DRAWDOWN, max_weight=0.3, n_samples=FAST
    ).weights
    assert weights.max() <= 0.3 + 1e-6
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def test_weights_are_valid_for_every_objective():
    panel = _panel()
    for objective in Objective:
        weights = optimize(panel, objective, n_samples=FAST).weights
        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert (weights >= -1e-9).all()


def test_weight_cap_is_respected():
    panel = _panel()
    for cap in (0.4, 0.5):
        for objective in Objective:
            weights = optimize(
                panel, objective, max_weight=cap, n_samples=FAST
            ).weights
            assert weights.max() <= cap + 1e-6


def test_capping_cannot_improve_the_objective():
    """A smaller feasible set cannot contain a better answer."""
    panel = _panel()
    free = optimize(panel, Objective.MIN_VOLATILITY, n_samples=FAST)
    capped = optimize(panel, Objective.MIN_VOLATILITY, max_weight=0.4, n_samples=FAST)
    assert capped.stats.volatility >= free.stats.volatility - 1e-9


def test_cap_below_equal_weight_is_rejected():
    panel = _panel()
    with pytest.raises(ValueError, match="max_weight"):
        optimize(panel, Objective.MIN_VOLATILITY, max_weight=0.2, n_samples=FAST)


# ---------------------------------------------------------------------------
# Near-optimal range
# ---------------------------------------------------------------------------

def test_near_optimal_contains_more_than_one_allocation():
    """The whole point: structurally different portfolios land within a
    fraction of a percent of each other, and reporting only the winner implies
    a precision the data does not support."""
    panel = _panel()
    result = optimize(panel, Objective.MAX_SHARPE, tolerance=0.05, n_samples=FAST)
    assert len(result.near_optimal) > 1


def test_looser_tolerance_admits_more_allocations():
    panel = _panel()
    tight = optimize(panel, Objective.MAX_SHARPE, tolerance=0.01, n_samples=FAST)
    loose = optimize(panel, Objective.MAX_SHARPE, tolerance=0.20, n_samples=FAST)
    assert len(loose.near_optimal) > len(tight.near_optimal)


def test_ranges_bracket_the_best_allocation():
    panel = _panel()
    ranges = optimize(
        panel, Objective.MAX_SHARPE, tolerance=0.10, n_samples=FAST
    ).ranges()
    assert (ranges["low"] <= ranges["best"] + 1e-9).all()
    assert (ranges["best"] <= ranges["high"] + 1e-9).all()


def test_tolerance_must_be_a_fraction():
    panel = _panel()
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="tolerance"):
            optimize(panel, Objective.MAX_SHARPE, tolerance=bad, n_samples=FAST)


# ---------------------------------------------------------------------------
# Consistency with the measurement layer
# ---------------------------------------------------------------------------

def test_reported_statistics_match_portfolio_stats():
    """Two routes to one number. The optimizer must not carry its own copy of
    the measurement logic."""
    panel = _panel()
    for objective in Objective:
        result = optimize(panel, objective, n_samples=FAST)
        direct = portfolio_stats(panel, result.weights)
        assert result.stats.volatility == pytest.approx(direct.volatility, rel=1e-12)
        assert result.stats.max_drawdown == pytest.approx(direct.max_drawdown, rel=1e-12)


def test_vectorised_sampling_agrees_with_the_measurement_layer():
    """The sampler computes statistics in bulk for speed. It must agree with
    the per-portfolio path exactly, or the search optimises the wrong thing."""
    from core.optimize import _measure_samples

    panel = _panel()
    samples = sample_allocations(len(panel.assets), 50, seed=3)
    measured = _measure_samples(panel, samples, None)

    for i in (0, 17, 49):
        weights = pd.Series(samples[i], index=panel.assets)
        direct = portfolio_stats(panel, weights)
        assert measured.loc[i, "volatility"] == pytest.approx(direct.volatility, rel=1e-9)
        assert measured.loc[i, "max_drawdown"] == pytest.approx(
            direct.max_drawdown, rel=1e-9
        )
        assert measured.loc[i, "sharpe"] == pytest.approx(direct.sharpe, rel=1e-9)
        assert measured.loc[i, "sortino"] == pytest.approx(direct.sortino, rel=1e-9)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_samples_are_valid_allocations():
    samples = sample_allocations(5, 500, seed=1)
    assert np.allclose(samples.sum(axis=1), 1.0, atol=1e-9)
    assert (samples >= -1e-12).all()


def test_samples_include_equal_weight():
    samples = sample_allocations(4, 200, seed=1)
    assert np.isclose(samples, 0.25).all(axis=1).any()


def test_samples_reach_the_corners():
    """Pure Dirichlet sampling rarely lands near a corner, and several
    objectives have their optimum there."""
    samples = sample_allocations(5, 2_000, seed=1)
    assert samples.max(axis=1).max() > 0.85


def test_capped_samples_respect_the_cap():
    samples = sample_allocations(5, 1_000, max_weight=0.3, seed=1)
    assert samples.max() <= 0.3 + 1e-6
    assert np.allclose(samples.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------

def test_frontier_is_monotonic_and_convex():
    panel = _panel()
    curve = efficient_frontier(panel, n_points=25)

    assert len(curve) >= 20
    assert curve["expected_return"].is_monotonic_increasing
    assert curve["volatility"].is_monotonic_increasing

    second_difference = np.diff(np.diff(curve["volatility"].to_numpy()))
    assert (second_difference > -1e-6).all()


def test_frontier_starts_at_minimum_variance():
    panel = _panel()
    curve = efficient_frontier(panel, n_points=20)
    floor = optimize(panel, Objective.MIN_VOLATILITY, n_samples=FAST)
    assert curve["volatility"].iloc[0] == pytest.approx(floor.stats.volatility, abs=1e-6)


def test_frontier_ends_at_the_best_asset():
    panel = _panel()
    curve = efficient_frontier(panel, n_points=20)
    assert curve["expected_return"].iloc[-1] == pytest.approx(
        panel.ann_return().max(), abs=1e-6
    )


def test_frontier_weights_are_valid():
    panel = _panel()
    curve = efficient_frontier(panel, n_points=15)
    assert np.allclose(curve[panel.assets].sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    assert (curve[panel.assets].to_numpy() >= -1e-9).all()


def test_frontier_respects_a_cap():
    panel = _panel()
    curve = efficient_frontier(panel, n_points=15, max_weight=0.5)
    assert curve[panel.assets].to_numpy().max() <= 0.5 + 1e-6


def test_cap_is_reached_exactly_not_approximately():
    """Water-filling should saturate the cap rather than stopping short --
    if it stopped short, the capped search would be exploring a smaller space
    than the user asked for."""
    for n_assets, cap in ((5, 0.30), (4, 0.40), (6, 0.25)):
        samples = sample_allocations(n_assets, 2_000, max_weight=cap, seed=1)
        assert samples.max() == pytest.approx(cap, abs=1e-9)
        assert np.allclose(samples.sum(axis=1), 1.0, atol=1e-9)


def test_naive_clip_and_renormalise_would_violate_the_cap():
    """Documents why the water-filling exists. Renormalising after clipping
    scales everything up, pushing capped assets back over the limit, and
    iterating never converges."""
    rng = np.random.default_rng(1)
    draws = rng.dirichlet(np.full(5, 0.15), size=500)
    cap = 0.3

    naive = np.minimum(draws, cap)
    naive = naive / naive.sum(axis=1, keepdims=True)
    assert naive.max() > cap + 1e-6

    proper = sample_allocations(5, 500, max_weight=cap, seed=1)
    assert proper.max() <= cap + 1e-9


# ---------------------------------------------------------------------------
# Degenerate objectives
# ---------------------------------------------------------------------------

def _degenerate_panel(n: int = 120) -> ReturnPanel:
    """A panel whose cash sleeve never falls below itself, so an all-cash
    allocation has zero downside deviation and an unbounded Sortino."""
    rng = np.random.default_rng(4)
    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": rng.normal(0.006, 0.046, n),
                "bonds": rng.normal(0.002, 0.013, n),
                "cash": np.full(n, 0.0015),
            },
            index=_dates(n),
        )
    )


def test_sampler_and_measurement_agree_on_degenerate_ratios():
    """The sampler drives the search while `portfolio_stats` reports the
    result. If they score the same allocation differently, the search is
    optimising something the output disagrees with -- silently."""
    from core.optimize import _measure_samples

    panel = _degenerate_panel()
    samples = np.array(
        [
            [0.0, 0.0, 1.0],  # all cash: zero downside deviation
            [0.5, 0.3, 0.2],
            [1.0, 0.0, 0.0],
        ]
    )
    measured = _measure_samples(panel, samples, None)

    for i in range(len(samples)):
        direct = portfolio_stats(panel, pd.Series(samples[i], index=panel.assets))
        for column, value in (("sortino", direct.sortino), ("sharpe", direct.sharpe)):
            sampled = measured.loc[i, column]
            if np.isinf(value):
                assert np.isinf(sampled) and np.sign(sampled) == np.sign(value)
            else:
                assert sampled == pytest.approx(value, rel=1e-9)


def test_optimizer_does_not_select_an_unbounded_objective():
    """A cash portfolio never falls below cash, so its Sortino is infinite.
    Arithmetically true, financially meaningless, and not an answer any user
    asked for. The search must skip it rather than crown it."""
    panel = _degenerate_panel()
    result = optimize(panel, Objective.MAX_SORTINO, n_samples=FAST)

    assert np.isfinite(result.value)
    assert result.weights["cash"] < 0.99


def test_near_optimal_is_empty_when_the_best_is_unbounded():
    """A range around infinity is not meaningful, so none is reported."""
    from core.optimize import _near_optimal

    measured = pd.DataFrame({"sortino": [np.inf, 0.5, 0.4]})
    kept = _near_optimal(measured, Objective.MAX_SORTINO, np.inf, 0.02)
    assert kept.empty


def test_the_reported_range_always_contains_the_best_allocation():
    """For an exactly-solved objective the optimum comes from the solver and
    the near-optimal set comes from sampling, so the optimum is not a member of
    that set. Without folding it in, the range can sit entirely to one side of
    the answer it claims to describe."""
    panel = _panel()
    for objective in Objective:
        for tolerance in (0.01, 0.05, 0.20):
            result = optimize(
                panel, objective, tolerance=tolerance, n_samples=FAST
            )
            ranges = result.ranges()
            assert (ranges["low"] <= ranges["best"] + 1e-12).all(), (
                f"{objective.value} at tolerance {tolerance}"
            )
            assert (ranges["best"] <= ranges["high"] + 1e-12).all()
            assert (ranges["spread"] >= -1e-12).all()
