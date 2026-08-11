"""Tests for core/panels.py.

Every assertion checks against something known independently of the
implementation: a hand-computed figure, an algebraic identity, or a property
that must hold by construction. Nothing asserts that the output equals
whatever the code currently produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.panels import (
    PriceHistory,
    ReturnPanel,
    blend_levels,
    levels_correlation_bias,
    splice_levels,
)


def _dates(n: int, start: str = "2006-10-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _walk(n: int, drift: float, vol: float, seed: int) -> np.ndarray:
    """Geometric random walk with independent increments."""
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def _history(n: int = 60, seed: int = 0) -> PriceHistory:
    return PriceHistory(
        pd.DataFrame(
            {
                "equity": _walk(n, 0.006, 0.04, seed),
                "bonds": _walk(n, 0.002, 0.011, seed + 1),
            },
            index=_dates(n),
        )
    )


# ---------------------------------------------------------------------------
# PriceHistory guards
# ---------------------------------------------------------------------------

def test_rejects_descending_index():
    """A reversed index negates every return with no other symptom."""
    df = pd.DataFrame({"a": np.linspace(100, 110, 10)}, index=_dates(10)[::-1])
    with pytest.raises(ValueError, match="sorted ascending"):
        PriceHistory(df)


def test_rejects_duplicate_dates():
    idx = pd.DatetimeIndex(["2006-10-31", "2006-10-31", "2006-11-30"])
    with pytest.raises(ValueError, match="duplicate dates"):
        PriceHistory(pd.DataFrame({"a": [100.0, 101.0, 102.0]}, index=idx))


def test_rejects_non_positive_levels_and_names_the_column():
    df = pd.DataFrame(
        {"good": [100.0, 101.0, 102.0], "bad": [100.0, 0.0, 102.0]}, index=_dates(3)
    )
    with pytest.raises(ValueError, match=r"Non-positive levels in \['bad'\]"):
        PriceHistory(df)


def test_rejects_non_datetime_index():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        PriceHistory(pd.DataFrame({"a": [1.0, 2.0]}))


def test_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        PriceHistory(pd.DataFrame(index=pd.DatetimeIndex([])))


# ---------------------------------------------------------------------------
# ReturnPanel guards
# ---------------------------------------------------------------------------

def test_return_panel_rejects_nans_and_names_the_column():
    df = pd.DataFrame(
        {"a": [0.01, 0.02, 0.03], "b": [0.01, np.nan, 0.03]}, index=_dates(3)
    )
    with pytest.raises(ValueError, match=r"NaNs in \['b'\]"):
        ReturnPanel(df)


def test_return_panel_rejects_impossible_returns():
    """A simple return at or below -100% means a splice was done wrong."""
    df = pd.DataFrame({"a": [0.01, -0.993, 0.02]}, index=_dates(3))
    ReturnPanel(df)  # -99.3% is legal, if alarming

    df_bad = pd.DataFrame({"a": [0.01, -1.0, 0.02]}, index=_dates(3))
    with pytest.raises(ValueError, match="chain-linking"):
        ReturnPanel(df_bad)


def test_return_panel_needs_two_observations():
    with pytest.raises(ValueError, match="at least 2"):
        ReturnPanel(pd.DataFrame({"a": [0.01]}, index=_dates(1)))


# ---------------------------------------------------------------------------
# Returns arithmetic
# ---------------------------------------------------------------------------

def test_returns_are_hand_computable():
    levels = pd.DataFrame({"a": [100.0, 110.0, 99.0]}, index=_dates(3))
    rets = PriceHistory(levels).returns().returns["a"]
    assert list(rets) == pytest.approx([0.10, -0.10])


def test_returns_drops_exactly_one_row():
    hist = _history(n=24)
    assert len(hist.returns()) == len(hist) - 1


def test_geometric_return_hand_computed():
    """+50% then -50%: growth factor 0.75 over two months, annualised."""
    panel = ReturnPanel(pd.DataFrame({"a": [0.5, -0.5]}, index=_dates(2)))
    assert panel.ann_return()["a"] == pytest.approx(0.75**6 - 1.0)


def test_geometric_is_below_arithmetic_when_variance_is_positive():
    panel = _history(n=120).returns()
    assert (panel.ann_return() < panel.arith_ann_return()).all()


def test_constant_return_makes_geometric_equal_arithmetic():
    """With zero variance the two definitions coincide exactly."""
    panel = ReturnPanel(pd.DataFrame({"a": [0.01] * 12}, index=_dates(12)))
    assert panel.ann_return()["a"] == pytest.approx(panel.arith_ann_return()["a"])
    assert panel.ann_return()["a"] == pytest.approx(1.01**12 - 1.0)


def test_annualised_vol_matches_closed_form():
    panel = _history(n=120).returns()
    expected = panel.returns["equity"].std(ddof=1) * np.sqrt(12)
    assert panel.ann_vol()["equity"] == pytest.approx(expected, rel=1e-12)


def test_cumulative_matches_compounded_returns():
    panel = _history(n=36).returns()
    final = panel.cumulative()["equity"].iloc[-1]
    assert final == pytest.approx((1 + panel.returns["equity"]).prod())


# ---------------------------------------------------------------------------
# The levels-versus-returns error
# ---------------------------------------------------------------------------

def test_price_history_cannot_produce_a_correlation():
    """The API must not offer this by accident."""
    hist = _history()
    assert not hasattr(hist, "corr")
    assert not hasattr(hist, "cov")


def test_levels_correlate_strongly_when_returns_do_not():
    """Independent random walks: near-zero return correlation, large level
    correlation. Asserted across many draws, since a single realisation can
    land anywhere by luck of the drift paths."""
    trials = 40
    level_corrs, return_corrs = [], []

    for k in range(trials):
        hist = PriceHistory(
            pd.DataFrame(
                {"a": _walk(300, 0.006, 0.04, 2 * k), "b": _walk(300, 0.006, 0.04, 2 * k + 1)},
                index=_dates(300),
            )
        )
        row = levels_correlation_bias(hist).iloc[0]
        level_corrs.append(row["corr_levels"])
        return_corrs.append(row["corr_returns"])

    level_corrs = np.array(level_corrs)
    return_corrs = np.array(return_corrs)

    # Increments are independent by construction.
    assert np.abs(return_corrs).max() < 0.20
    assert abs(np.median(return_corrs)) < 0.05

    # Levels are dominated by drift. Sign depends on whether the paths trended
    # together or apart, so the claim is about magnitude.
    assert np.median(np.abs(level_corrs)) > 0.70
    assert (np.abs(level_corrs) > np.abs(return_corrs)).all()


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------

def test_splice_preserves_each_segment_returns():
    idx = _dates(24)
    older = pd.Series(150.0 * np.cumprod(np.full(14, 1.001)), index=idx[:14])
    newer = pd.Series(1.0 * np.cumprod(np.full(10, 1.002)), index=idx[14:])

    out = splice_levels(newer, older)
    rets = out.pct_change().dropna()

    assert len(out) == 24
    assert rets.iloc[:12].round(10).eq(0.001).all()
    assert rets.iloc[-9:].round(10).eq(0.002).all()
    assert abs(rets).max() < 0.01, "no fabricated jump at the junction"


def test_naive_concatenation_would_fabricate_a_crash():
    """Documents the failure the splice exists to prevent."""
    idx = _dates(24)
    older = pd.Series(np.full(14, 151.0), index=idx[:14])
    newer = pd.Series(np.full(10, 1.0), index=idx[14:])
    naive = pd.concat([older, newer]).pct_change().dropna()
    assert naive.min() < -0.99


def test_splice_with_overlap_anchors_on_shared_date():
    idx = _dates(20)
    older = pd.Series(np.linspace(50.0, 69.0, 20), index=idx)
    newer = pd.Series(np.linspace(100.0, 118.0, 10), index=idx[10:])

    out = splice_levels(newer, older)
    assert out.loc[idx[10]] == pytest.approx(newer.iloc[0])
    assert len(out) == 20


def test_splice_rejects_wrong_order():
    idx = _dates(20)
    a = pd.Series(np.linspace(100, 120, 10), index=idx[:10])
    b = pd.Series(np.linspace(100, 120, 10), index=idx[10:])
    with pytest.raises(ValueError, match="must start before"):
        splice_levels(a, b)


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def test_blend_of_identical_components_equals_the_component():
    idx = _dates(30)
    series = _walk(30, 0.005, 0.03, seed=4)
    frame = pd.DataFrame({"x": series, "y": series * 7.0}, index=idx)

    blended = blend_levels(frame, {"x": 0.6, "y": 0.4})
    expected = pd.Series(series / series[0], index=idx)
    assert np.allclose(blended.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_blend_return_is_the_weighted_average_of_component_returns():
    idx = _dates(30)
    frame = pd.DataFrame(
        {"x": _walk(30, 0.006, 0.04, 5), "y": _walk(30, 0.003, 0.02, 6)}, index=idx
    )
    weights = {"x": 0.55, "y": 0.45}

    blended = blend_levels(frame, weights).pct_change().dropna()
    manual = frame.pct_change().dropna().mul(pd.Series(weights), axis=1).sum(axis=1)
    assert np.allclose(blended.to_numpy(), manual.to_numpy(), atol=1e-12)


def test_blend_weights_must_sum_to_one():
    frame = pd.DataFrame(
        {"x": _walk(10, 0.005, 0.03, 7), "y": _walk(10, 0.005, 0.03, 8)},
        index=_dates(10),
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        blend_levels(frame, {"x": 0.6, "y": 0.5})


def test_blend_rejects_unknown_component():
    frame = pd.DataFrame({"x": _walk(10, 0.005, 0.03, 9)}, index=_dates(10))
    with pytest.raises(KeyError, match="missing"):
        blend_levels(frame, {"x": 0.5, "missing": 0.5})


# ---------------------------------------------------------------------------
# Windowing and selection
# ---------------------------------------------------------------------------

def test_binding_asset_identifies_the_shortest_history():
    idx = _dates(24)
    long = pd.Series(np.linspace(100, 150, 24), index=idx)
    short = pd.Series(np.linspace(100, 120, 24), index=idx)
    short.iloc[:10] = np.nan

    hist = PriceHistory(pd.DataFrame({"long": long, "short": short}))
    name, first = hist.binding_asset()
    assert name == "short"
    assert first == idx[10]


def test_common_window_trims_to_the_intersection():
    idx = _dates(24)
    long = pd.Series(np.linspace(100, 150, 24), index=idx)
    short = pd.Series(np.linspace(100, 120, 24), index=idx)
    short.iloc[:10] = np.nan

    trimmed = PriceHistory(pd.DataFrame({"long": long, "short": short})).common_window()
    assert len(trimmed) == 14
    assert trimmed.levels.index[0] == idx[10]


def test_empty_column_is_reported_by_name():
    """A failed fetch must be diagnosed as a failed fetch, not as a confusing
    complaint about observation counts three files later."""
    idx = _dates(24)
    hist = PriceHistory(
        pd.DataFrame(
            {"equity": np.linspace(100, 150, 24), "private_equity": [np.nan] * 24},
            index=idx,
        )
    )
    with pytest.raises(ValueError, match=r"no data at all.*private_equity"):
        hist.common_window()


def test_insufficient_overlap_reports_each_asset_coverage():
    idx = _dates(24)
    a = pd.Series(np.linspace(100, 150, 24), index=idx)
    b = pd.Series(np.linspace(100, 120, 24), index=idx)
    a.iloc[12:] = np.nan
    b.iloc[:12] = np.nan

    hist = PriceHistory(pd.DataFrame({"a": a, "b": b}))
    with pytest.raises(ValueError, match="share only"):
        hist.common_window()


def test_select_restricts_assets_and_rejects_unknown():
    panel = _history(n=36).returns()
    assert panel.select(["bonds"]).assets == ["bonds"]
    with pytest.raises(KeyError, match="gold"):
        panel.select(["bonds", "gold"])


def test_between_slices_inclusively():
    panel = _history(n=60).returns()
    sub = panel.between("2008-01-01", "2008-12-31")
    assert len(sub) == 12
    assert sub.start.year == 2008 and sub.end.year == 2008


def test_between_rejects_a_window_with_too_little_data():
    panel = _history(n=60).returns()
    with pytest.raises(ValueError, match="observations"):
        panel.between("2020-01-01", "2020-02-01")


def test_month_end_resampling_keeps_the_last_level():
    """61 days from 1 October covers October (31 days) and November (30),
    so exactly two month-ends, and each takes that month's final value."""
    daily = pd.DataFrame(
        {"a": np.arange(1.0, 62.0)}, index=pd.date_range("2006-10-01", periods=61)
    )
    monthly = PriceHistory(daily).to_month_end()
    assert len(monthly) == 2
    assert monthly.levels["a"].iloc[0] == 31.0  # 31 Oct
    assert monthly.levels["a"].iloc[1] == 61.0  # 30 Nov
    assert list(monthly.levels.index.day) == [31, 30]


def test_incomplete_final_month_is_dropped():
    """62 days from 1 October reaches 1 December. That single December day is
    a partial month and must not be reported as a full one -- otherwise a
    dataset built mid-month carries an 11-day return labelled as a monthly
    return, understating that month and every figure derived from it."""
    daily = pd.DataFrame(
        {"a": np.arange(1.0, 63.0)}, index=pd.date_range("2006-10-01", periods=62)
    )
    monthly = PriceHistory(daily).to_month_end()

    assert len(monthly) == 2
    assert monthly.levels.index[-1].month == 11
    assert monthly.levels["a"].iloc[-1] == 61.0


def test_incomplete_final_month_can_be_kept_explicitly():
    daily = pd.DataFrame(
        {"a": np.arange(1.0, 63.0)}, index=pd.date_range("2006-10-01", periods=62)
    )
    monthly = PriceHistory(daily).to_month_end(drop_incomplete=False)
    assert len(monthly) == 3
    assert monthly.levels["a"].iloc[-1] == 62.0


def test_month_ending_on_a_weekend_is_kept():
    """31 May 2026 is a Sunday, so the last trading day is Friday 29 May. That
    month is complete and must survive -- the tolerance exists for exactly
    this, and being too strict would silently discard a real month."""
    trading_days = pd.bdate_range("2026-04-01", "2026-05-29")
    daily = pd.DataFrame(
        {"a": np.linspace(100.0, 120.0, len(trading_days))}, index=trading_days
    )
    monthly = PriceHistory(daily).to_month_end()

    assert monthly.levels.index[-1].month == 5
    assert len(monthly) == 2


def test_a_long_holiday_gap_still_drops_a_genuinely_partial_month():
    """The tolerance must not be so generous that a half-finished month
    survives. Data ending 10 August is 21 days short and has to go."""
    daily = pd.DataFrame(
        {"a": np.linspace(100.0, 130.0, 71)},
        index=pd.date_range("2026-06-01", "2026-08-10"),
    )
    monthly = PriceHistory(daily).to_month_end()

    assert monthly.levels.index[-1].month == 7
    assert len(monthly) == 2
