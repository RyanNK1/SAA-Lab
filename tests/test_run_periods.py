"""Tests for scripts/run_periods.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core.config import Objective  # noqa: E402
from core.panels import ReturnPanel  # noqa: E402
from core.periods import (  # noqa: E402
    compare_periods,
    consensus_allocation,
    cross_period_performance,
    hindsight_premium,
    resolve_periods,
)
from core.sleeve import build_sleeve  # noqa: E402


def _panel(n: int = 245, seed: int = 3) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
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
    # A crisis, so the named regimes are genuinely different from each other.
    frame.iloc[18:35, [0, 2]] -= 0.06
    return ReturnPanel(frame)


def test_named_regimes_resolve_against_a_full_length_panel():
    """The seven windows should mostly survive against 20 years of data. If
    they did not, the comparison would silently reduce to two or three."""
    sleeved = build_sleeve(_panel(), 0.5)
    periods = resolve_periods(sleeved)
    assert len(periods) >= 5


def test_comparison_runs_end_to_end_on_a_sleeved_panel():
    sleeved = build_sleeve(_panel(), 0.5)
    periods = resolve_periods(sleeved)
    table, results = compare_periods(sleeved, periods, n_samples=1_000)

    assert len(table) == len(periods)
    assert np.allclose(table[sleeved.assets].sum(axis=1).to_numpy(), 1.0, atol=1e-8)


def test_the_answer_genuinely_changes_between_regimes():
    """If every period gave the same allocation, the whole feature would be
    pointless. The fixture contains a crisis for exactly this reason."""
    sleeved = build_sleeve(_panel(), 0.5)
    periods = resolve_periods(sleeved)
    table, _ = compare_periods(sleeved, periods, n_samples=1_000)

    spreads = table[sleeved.assets].max() - table[sleeved.assets].min()
    assert spreads.max() > 0.30


def test_hindsight_premium_is_positive_across_real_regimes():
    """Every period's own allocation beats the others in that period, because
    it was chosen knowing what happened. The size of that gap is what a
    single-period result overstates by."""
    sleeved = build_sleeve(_panel(), 0.5)
    periods = resolve_periods(sleeved)
    _, results = compare_periods(sleeved, periods, n_samples=1_000)

    premium = hindsight_premium(cross_period_performance(sleeved, results, periods))
    assert (premium["premium"] >= -1e-6).all()
    assert premium["premium"].mean() > 0.0


def test_an_allocation_can_lose_badly_outside_its_own_period():
    """The finding the matrix exists to surface: a hindsight winner is not a
    durable answer."""
    sleeved = build_sleeve(_panel(), 0.5)
    periods = resolve_periods(sleeved)
    _, results = compare_periods(sleeved, periods, n_samples=1_000)
    matrix = cross_period_performance(sleeved, results, periods)

    off_diagonal = matrix.to_numpy()[~np.eye(len(matrix), dtype=bool)]
    diagonal = np.diag(matrix.to_numpy())
    assert off_diagonal.min() < diagonal.min()


def test_consensus_is_valid_and_moderate():
    sleeved = build_sleeve(_panel(), 0.5)
    table, _ = compare_periods(sleeved, resolve_periods(sleeved), n_samples=1_000)
    consensus = consensus_allocation(table, sleeved.assets)

    assert consensus.sum() == pytest.approx(1.0)
    assert (consensus >= -1e-12).all()
    assert consensus.max() < 1.0, "an average should not be a corner solution"
