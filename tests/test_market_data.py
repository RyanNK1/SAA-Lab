"""Tests for core/market_data.py.

The fetch itself cannot be tested here -- it needs a network. What *can* be
tested is everything built on top of the fetched data: the equity splice, the
cash compounding, the empty-column guard, and the validation report. Those are
where the errors that matter live, because a network failure is loud while a
splice error is silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.market_data import (
    EQUITY_BLEND_TICKERS,
    EQUITY_BLEND_WEIGHTS,
    _reject_empty_columns,
    build_equity_series,
    compound_rate_to_index,
    validate,
)
from core.panels import PriceHistory


def _days(n: int, start: str = "2004-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


def _walk(n: int, drift: float, vol: float, seed: int, base: float = 100.0):
    rng = np.random.default_rng(seed)
    return base * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def _levels_frame(n: int = 800, acwi_starts: int = 500) -> pd.DataFrame:
    """Three blend components spanning the whole period; ACWI starting later.

    Mirrors the real shape: SPY/EFA/EEM run throughout, ACWI appears partway.
    """
    idx = _days(n)
    frame = pd.DataFrame(
        {
            "us": _walk(n, 0.0004, 0.010, 1),
            "developed_ex_us": _walk(n, 0.0003, 0.011, 2),
            "emerging": _walk(n, 0.0005, 0.015, 3),
            "acwi": _walk(n, 0.0004, 0.010, 4, base=40.0),
        },
        index=idx,
    )
    frame.loc[idx[:acwi_starts], "acwi"] = np.nan
    return frame


# ---------------------------------------------------------------------------
# Configuration sanity
# ---------------------------------------------------------------------------

def test_blend_weights_sum_to_one():
    assert sum(EQUITY_BLEND_WEIGHTS.values()) == pytest.approx(1.0)


def test_blend_weights_cover_every_blend_ticker():
    assert set(EQUITY_BLEND_WEIGHTS) == set(EQUITY_BLEND_TICKERS)


# ---------------------------------------------------------------------------
# The equity splice
# ---------------------------------------------------------------------------

def test_equity_series_spans_the_full_period():
    """The whole point: coverage back to before ACWI existed."""
    frame = _levels_frame()
    equity = build_equity_series(frame)
    assert equity.index[0] == frame.index[0]
    assert equity.index[-1] == frame.index[-1]
    assert equity.notna().all()


def test_equity_matches_acwi_returns_after_the_junction():
    """From the junction onward the series must be ACWI exactly, not an
    approximation of it."""
    frame = _levels_frame()
    equity = build_equity_series(frame)
    acwi = frame["acwi"].dropna()

    equity_rets = equity.pct_change().loc[acwi.index[1:]]
    acwi_rets = acwi.pct_change().iloc[1:]
    assert np.allclose(equity_rets.to_numpy(), acwi_rets.to_numpy(), atol=1e-12)


def test_equity_matches_the_blend_before_the_junction():
    frame = _levels_frame()
    equity = build_equity_series(frame)
    junction = frame["acwi"].first_valid_index()

    manual = (
        frame[list(EQUITY_BLEND_WEIGHTS)]
        .pct_change()
        .iloc[1:]
        .mul(pd.Series(EQUITY_BLEND_WEIGHTS), axis=1)
        .sum(axis=1)
    )
    equity_rets = equity.pct_change().dropna()
    before = equity_rets.index < junction

    assert np.allclose(
        equity_rets[before].to_numpy(), manual.loc[equity_rets.index[before]].to_numpy(),
        atol=1e-12,
    )


def test_splice_creates_no_artificial_jump():
    """The failure mode this construction exists to avoid. ACWI is built at a
    deliberately different scale (40 vs 100) so a naive join would show it."""
    frame = _levels_frame()
    equity = build_equity_series(frame)
    rets = equity.pct_change().dropna()

    junction = frame["acwi"].first_valid_index()
    window = rets.loc[
        (rets.index >= junction - pd.Timedelta(days=5))
        & (rets.index <= junction + pd.Timedelta(days=5))
    ]
    assert abs(window).max() < 0.15, "a scale mismatch leaked into the returns"
    assert rets.min() > -0.5


def test_naive_concatenation_would_have_produced_a_crash():
    """Documents what the chain-link prevents, so the reason stays visible."""
    frame = _levels_frame()
    acwi = frame["acwi"].dropna()
    blend_before = frame["us"][frame.index < acwi.index[0]]
    naive = pd.concat([blend_before, acwi]).pct_change().dropna()
    assert naive.min() < -0.5


def test_equity_construction_reports_missing_inputs():
    frame = _levels_frame().drop(columns=["emerging"])
    with pytest.raises(KeyError, match="emerging"):
        build_equity_series(frame)


def test_equity_construction_rejects_empty_acwi():
    frame = _levels_frame()
    frame["acwi"] = np.nan
    with pytest.raises((RuntimeError, ValueError)):
        build_equity_series(frame)


# ---------------------------------------------------------------------------
# Cash
# ---------------------------------------------------------------------------

def test_constant_rate_compounds_to_that_rate():
    """A flat 5% for a year must grow the index by very close to 5%."""
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    rate = pd.Series(0.05, index=idx)
    index = compound_rate_to_index(rate)
    assert index.iloc[-1] == pytest.approx(1.05, rel=0.002)


def test_cash_index_rises_while_rates_are_positive():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    rng = np.random.default_rng(11)
    rate = pd.Series(np.abs(rng.normal(0.03, 0.01, 400)), index=idx)
    index = compound_rate_to_index(rate)

    assert index.iloc[0] == 1.0
    assert index.diff().dropna().min() >= 0, "a positive rate cannot lose money"


def test_negative_rates_make_the_cash_index_fall():
    """US T-bills traded below zero in late 2015 and March 2020. Cash losing
    a little value in those windows is correct history, not an error."""
    idx = pd.date_range("2020-02-01", periods=90, freq="D")
    rate = pd.Series(
        np.concatenate([np.full(30, 0.015), np.full(30, -0.001), np.full(30, 0.010)]),
        index=idx,
    )
    index = compound_rate_to_index(rate)

    negative_stretch = index.iloc[31:60]
    assert negative_stretch.diff().dropna().max() <= 0, "should drift down"
    assert index.iloc[-1] > 1.0, "recovers once rates turn positive again"


def test_zero_rate_leaves_the_index_flat():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    index = compound_rate_to_index(pd.Series(0.0, index=idx))
    assert index.iloc[-1] == pytest.approx(1.0)


def test_higher_rates_compound_faster():
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    low = compound_rate_to_index(pd.Series(0.01, index=idx)).iloc[-1]
    high = compound_rate_to_index(pd.Series(0.05, index=idx)).iloc[-1]
    assert high > low


def test_compounding_handles_weekday_gaps():
    """Real rate series skip weekends. The day count must bridge them rather
    than treating each observation as one day."""
    idx = pd.date_range("2020-01-01", periods=260, freq="B")
    index = compound_rate_to_index(pd.Series(0.05, index=idx))
    assert index.iloc[-1] == pytest.approx(1.05, rel=0.01)


def test_rate_series_needs_two_observations():
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    with pytest.raises(ValueError, match="at least 2"):
        compound_rate_to_index(pd.Series(0.05, index=idx))


# ---------------------------------------------------------------------------
# Fetch guards
# ---------------------------------------------------------------------------

def test_empty_column_is_named_with_its_ticker():
    frame = pd.DataFrame(
        {"equity": [1.0, 2.0], "private_equity": [np.nan, np.nan]}, index=_days(2)
    )
    with pytest.raises(RuntimeError, match=r"private_equity \(PSP\)"):
        _reject_empty_columns(frame, {"equity": "ACWI", "private_equity": "PSP"})


def test_full_columns_pass_the_guard():
    frame = pd.DataFrame({"equity": [1.0, 2.0]}, index=_days(2))
    _reject_empty_columns(frame, {"equity": "ACWI"})


def test_partially_missing_column_is_allowed():
    """Gaps are normal -- an asset starting later is the expected case. Only a
    wholly empty column indicates a failed fetch."""
    frame = pd.DataFrame({"a": [np.nan, np.nan, 3.0]}, index=_days(3))
    _reject_empty_columns(frame, {"a": "AAA"})


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def test_validation_reports_one_row_per_asset():
    idx = pd.date_range("2006-10-31", periods=60, freq="ME")
    hist = PriceHistory(
        pd.DataFrame(
            {"equity": _walk(60, 0.006, 0.04, 7), "cash": _walk(60, 0.001, 0.0005, 8)},
            index=idx,
        )
    )
    report = validate(hist)
    assert list(report.index) == ["equity", "cash"]
    assert report["n_missing"].sum() == 0


def test_validation_counts_stale_months():
    """A repeated price produces a zero return -- the stale-quote signature."""
    idx = pd.date_range("2006-10-31", periods=12, freq="ME")
    levels = np.linspace(100.0, 120.0, 12)
    levels[5] = levels[4]
    hist = PriceHistory(pd.DataFrame({"a": levels}, index=idx))
    assert validate(hist).loc["a", "zero_return_months"] == 1


def test_validation_surfaces_a_bad_splice():
    """min_return is the detector: a wrong chain-link shows up as one
    catastrophic month."""
    idx = pd.date_range("2006-10-31", periods=12, freq="ME")
    levels = np.concatenate([np.full(6, 151.0), np.full(6, 1.0)])
    hist = PriceHistory(pd.DataFrame({"a": levels}, index=idx))
    assert validate(hist).loc["a", "min_return"] < -0.99


def test_rate_floor_rejects_an_implausible_quote():
    """A deeply negative rate is a bad quote or the wrong series, not a market.
    Tested via the constant so the guard's intent stays documented."""
    from core.market_data import MIN_PLAUSIBLE_RATE

    assert -0.05 < MIN_PLAUSIBLE_RATE < 0.0, (
        "the floor must sit below real negative-rate episodes (a few bps) but "
        "above anything that would indicate a data error"
    )
