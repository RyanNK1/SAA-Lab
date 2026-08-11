"""Tests for core/sleeve.py.

The endpoint tests are the important ones: a slider at 100% gold must
reproduce the gold series *exactly*, not approximately. If that holds and the
interior is a correct weighted average, the sleeve is right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.panels import ReturnPanel
from core.sleeve import (
    COMMODITIES_EX_GOLD,
    GOLD,
    SLEEVE,
    SleeveSpec,
    build_sleeve,
    sleeve_components,
    sleeve_sensitivity,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2006-03-31", periods=n, freq="ME")


def _panel(n: int = 120, seed: int = 0) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    return ReturnPanel(
        pd.DataFrame(
            {
                "equity": rng.normal(0.006, 0.046, n),
                "fixed_income": rng.normal(0.002, 0.013, n),
                GOLD: rng.normal(0.008, 0.050, n),
                COMMODITIES_EX_GOLD: rng.normal(0.002, 0.054, n),
                "cash": np.abs(rng.normal(0.0012, 0.0004, n)),
            },
            index=_dates(n),
        )
    )


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def test_spec_rejects_weights_outside_the_slider_range():
    for bad in (-0.01, 1.01, 2.0, -1.0):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SleeveSpec(bad)


def test_spec_accepts_both_endpoints():
    assert SleeveSpec(0.0).is_pure_ex_gold
    assert SleeveSpec(1.0).is_pure_gold


def test_weights_are_complementary():
    spec = SleeveSpec(0.35)
    assert spec.gold_weight + spec.ex_gold_weight == pytest.approx(1.0)


def test_describe_is_readable():
    assert SleeveSpec(0.6).describe() == "60% gold / 40% ex-gold"


# ---------------------------------------------------------------------------
# Endpoints -- exact reproduction
# ---------------------------------------------------------------------------

def test_full_gold_reproduces_the_gold_series_exactly():
    panel = _panel()
    built = build_sleeve(panel, 1.0)
    assert np.allclose(
        built.returns[SLEEVE].to_numpy(), panel.returns[GOLD].to_numpy(), atol=1e-15
    )


def test_full_ex_gold_reproduces_that_series_exactly():
    panel = _panel()
    built = build_sleeve(panel, 0.0)
    assert np.allclose(
        built.returns[SLEEVE].to_numpy(),
        panel.returns[COMMODITIES_EX_GOLD].to_numpy(),
        atol=1e-15,
    )


def test_endpoint_statistics_match_the_component_exactly():
    """Not just the series -- the derived statistics too."""
    panel = _panel()
    built = build_sleeve(panel, 1.0)
    assert built.ann_return()[SLEEVE] == pytest.approx(
        panel.ann_return()[GOLD], rel=1e-12
    )
    assert built.ann_vol()[SLEEVE] == pytest.approx(panel.ann_vol()[GOLD], rel=1e-12)


# ---------------------------------------------------------------------------
# Interior
# ---------------------------------------------------------------------------

def test_sleeve_return_is_the_weighted_average_of_components():
    panel = _panel()
    w = 0.35
    built = build_sleeve(panel, w)
    manual = (
        panel.returns[GOLD] * w + panel.returns[COMMODITIES_EX_GOLD] * (1 - w)
    )
    assert np.allclose(built.returns[SLEEVE].to_numpy(), manual.to_numpy(), atol=1e-15)


def test_mixed_sleeve_volatility_sits_below_the_weighted_average():
    """Diversification within the sleeve. Unless the components correlate
    perfectly, mixing them must reduce volatility relative to a naive weighted
    average of the two standalone volatilities."""
    panel = _panel()
    vols = panel.ann_vol()
    naive = 0.5 * vols[GOLD] + 0.5 * vols[COMMODITIES_EX_GOLD]
    actual = build_sleeve(panel, 0.5).ann_vol()[SLEEVE]
    assert actual < naive


def test_sleeve_statistics_move_monotonically_between_endpoints():
    """No slider position may produce a return outside the range its two
    components span -- that would mean the arithmetic is wrong."""
    panel = _panel()
    low = build_sleeve(panel, 0.0).ann_return()[SLEEVE]
    high = build_sleeve(panel, 1.0).ann_return()[SLEEVE]

    for w in (0.1, 0.25, 0.5, 0.75, 0.9):
        mid = build_sleeve(panel, w).ann_return()[SLEEVE]
        assert min(low, high) <= mid <= max(low, high)


# ---------------------------------------------------------------------------
# Panel structure
# ---------------------------------------------------------------------------

def test_sleeve_replaces_both_components():
    panel = _panel()
    built = build_sleeve(panel, 0.5)
    assert SLEEVE in built.assets
    assert GOLD not in built.assets
    assert COMMODITIES_EX_GOLD not in built.assets


def test_asset_count_drops_by_exactly_one():
    panel = _panel()
    assert len(build_sleeve(panel, 0.5).assets) == len(panel.assets) - 1


def test_other_assets_are_untouched():
    panel = _panel()
    built = build_sleeve(panel, 0.7)
    for asset in ("equity", "fixed_income", "cash"):
        assert np.allclose(
            built.returns[asset].to_numpy(), panel.returns[asset].to_numpy(), atol=1e-15
        )


def test_column_order_is_stable_across_slider_positions():
    """Output ordering must not shift as the slider moves, or every table and
    chart downstream reshuffles for no reason."""
    panel = _panel()
    assert build_sleeve(panel, 0.1).assets == build_sleeve(panel, 0.9).assets


def test_index_is_preserved():
    panel = _panel()
    assert build_sleeve(panel, 0.5).returns.index.equals(panel.returns.index)


# ---------------------------------------------------------------------------
# Partial and absent components
# ---------------------------------------------------------------------------

def test_panel_without_either_component_is_returned_unchanged():
    """A user who deselected commodities has no sleeve to build. That is a
    legitimate request, not an error."""
    panel = _panel()
    trimmed = panel.select(["equity", "fixed_income", "cash"])
    built = build_sleeve(trimmed, 0.5)
    assert built.assets == trimmed.assets


def test_single_component_is_renamed_to_the_sleeve():
    """With only gold selected, the sleeve is gold. Downstream code should see
    a consistent name rather than having to special-case it."""
    panel = _panel().select(["equity", GOLD, "cash"])
    built = build_sleeve(panel, 0.5)
    assert SLEEVE in built.assets
    assert GOLD not in built.assets
    assert np.allclose(
        built.returns[SLEEVE].to_numpy(),
        panel.returns[GOLD].to_numpy(),
        atol=1e-15,
    )


def test_rejects_a_panel_that_already_has_a_sleeve_column():
    panel = _panel()
    renamed = panel.returns.rename(columns={"equity": SLEEVE})
    with pytest.raises(ValueError, match="already has a column"):
        build_sleeve(ReturnPanel(renamed), 0.5)


def test_components_helper_reports_what_is_present():
    panel = _panel()
    assert sleeve_components(panel) == [GOLD, COMMODITIES_EX_GOLD]
    assert sleeve_components(panel.select(["equity", GOLD])) == [GOLD]
    assert sleeve_components(panel.select(["equity"])) == []


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def test_sensitivity_spans_the_full_slider():
    panel = _panel()
    table = sleeve_sensitivity(panel, steps=11)
    assert len(table) == 11
    assert table["gold_weight"].iloc[0] == 0.0
    assert table["gold_weight"].iloc[-1] == 1.0


def test_sensitivity_endpoints_match_the_components():
    panel = _panel()
    table = sleeve_sensitivity(panel, steps=5)
    assert table["ann_return"].iloc[-1] == pytest.approx(
        panel.ann_return()[GOLD], rel=1e-12
    )
    assert table["ann_return"].iloc[0] == pytest.approx(
        panel.ann_return()[COMMODITIES_EX_GOLD], rel=1e-12
    )


def test_sensitivity_reports_correlation_with_every_other_asset():
    panel = _panel()
    table = sleeve_sensitivity(panel, steps=3)
    for asset in ("equity", "fixed_income", "cash"):
        assert f"corr_{asset}" in table.columns


def test_sensitivity_shows_correlation_changing_with_the_slider():
    """The point of the whole feature: 'commodities' is not one thing. Its
    relationship with equities depends on the slider."""
    panel = _panel(seed=3)
    table = sleeve_sensitivity(panel, steps=11)
    spread = table["corr_equity"].max() - table["corr_equity"].min()
    assert spread > 0.01, "the sleeve's character should vary with the slider"


def test_sensitivity_requires_both_components():
    panel = _panel().select(["equity", GOLD])
    with pytest.raises(ValueError, match="needs both"):
        sleeve_sensitivity(panel)
