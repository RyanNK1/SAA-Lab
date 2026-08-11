"""Fetching and construction of the asset-class price series.

NOT VERIFIED IN THE AUTHORING SANDBOX. Every other module in this project was
executed before being committed; this one could not be, because the sandbox
blocks outbound requests to the price source. The logic is written carefully
but is unrun. Treat the first local execution as the real test, and read the
validation report rather than trusting the shapes.

Proxies, and how each differs from the thing it represents:

    bucket           proxy         caveat
    ---------------- ------------- --------------------------------------------
    equity           ACWI          Only exists from 2008-03. Before that, a
                                   blend of SPY/EFA/EEM chain-linked behind it
    fixed_income     AGG           US aggregate, NOT global. No non-USD exposure
    private_equity   IWM           US small cap (Russell 2000), used as a
                                   replication proxy for private equity --
                                   see the note below
    gold             GLD           Physical gold trust, net of 0.40% fee
    commodities_ex_gold DBC        Broad futures basket. Carries roll yield
                                   effects that spot commodity prices do not
    cash             ^IRX          13-week T-bill *discount* rate, compounded
                                   here into an index

The binding constraint on history is DBC (2006-02), which is why the dataset
starts there. That window includes the full run-up to the financial crisis,
the crash, and everything since.

On the private equity sleeve
---------------------------
Nothing freely available measures institutional private equity. The two
candidates both compromise, in opposite directions:

  Listed PE vehicles (PSP and similar) hold publicly traded private equity
  *managers* -- leveraged financial holding companies, not the underlying
  deals. PSP launched at the 2006 peak, fell roughly 70% in the crash, and
  carries a ~1.4-1.8% expense ratio, so its realised return since inception is
  about 2.6% a year. That figure is correct and it is not what an allocator
  means by "private equity"; institutional buyout funds returned roughly
  10-13% net over the same period.

  Small-cap equity has no such distortion and returns in a plausible range.
  It is also defensible on its merits: there is a substantial literature
  arguing private equity returns are largely replicable with small-cap value
  plus leverage (Stafford, 2017). This is the proxy the original Excel study
  used.

Small cap is used here, and the honest caveat is that it correlates heavily
with public equity -- around 0.9 in monthly returns. So the "private equity"
sleeve is not the diversifier its label suggests. The risk contribution table
is built precisely to make that visible rather than let the label do the
talking.

Note also that institutional PE reports appraisal-based valuations, which
smooth returns and understate both volatility and correlation with equities.
A daily-marked proxy will always look riskier than an LP's own statements.
That is the proxy being honest, not wrong.
"""

from __future__ import annotations

import pandas as pd

from core.panels import PriceHistory, blend_levels, splice_levels

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

# Assets fetched directly, one ticker each.
DIRECT_TICKERS: dict[str, str] = {
    "fixed_income": "AGG",
    "private_equity": "IWM",
    "gold": "GLD",
    "commodities_ex_gold": "DBC",
}

# Equity is constructed rather than fetched directly.
EQUITY_TICKER = "ACWI"
EQUITY_BLEND_TICKERS: dict[str, str] = {
    "us": "SPY",
    "developed_ex_us": "EFA",
    "emerging": "EEM",
}
# Fixed weights approximating MSCI ACWI's composition. Fixed rather than
# time-varying: the accuracy gain from tracking actual market-cap drift is
# small, and the added complexity would be hard to verify.
EQUITY_BLEND_WEIGHTS: dict[str, float] = {
    "us": 0.55,
    "developed_ex_us": 0.33,
    "emerging": 0.12,
}

CASH_TICKER = "^IRX"

# Every ticker touched, for a single sequential fetch.
ALL_TICKERS: dict[str, str] = {
    **DIRECT_TICKERS,
    "acwi": EQUITY_TICKER,
    **EQUITY_BLEND_TICKERS,
}

DEFAULT_START = "2000-01-01"

# Floor for the cash rate. Small negatives are genuine -- US T-bills traded
# below zero in late 2015 and March 2020 -- so the guard sits well below
# anything historically observed and catches bad quotes, not real markets.
MIN_PLAUSIBLE_RATE = -0.01


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_levels(
    tickers: dict[str, str],
    start: str = DEFAULT_START,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close levels. Keys of `tickers` become column names.

    Two settings are deliberate and should not be removed:

    `auto_adjust=True` is passed explicitly rather than relied on as a default.
    It has changed between library versions, and unadjusted prices insert a
    fabricated crash at every split and drop every dividend.

    `threads=False` disables the parallel downloader. Its worker threads
    contend over a single small cache, and under contention a ticker fails with
    a database-lock error and silently returns an all-NaN column. Sequential
    fetching is a few seconds slower and does not have that failure mode.
    """
    import yfinance as yf

    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(
            "The price source returned nothing at all. Usual causes: no "
            "network, rate limiting (common from cloud hosts and after "
            "repeated pulls), or every ticker being wrong. If a previous run "
            "succeeded minutes ago, rate limiting is the likely cause -- wait "
            "and retry."
        )

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    inverted = {v: k for k, v in tickers.items()}
    close = close.rename(columns=inverted)

    missing = [name for name in tickers if name not in close.columns]
    if missing:
        raise RuntimeError(f"No column returned for {missing}")

    close = close[list(tickers)].sort_index()
    _reject_empty_columns(close, tickers)
    return close


def _reject_empty_columns(close: pd.DataFrame, tickers: dict[str, str]) -> None:
    """Fail immediately, naming the asset, if a ticker returned nothing.

    Without this the empty column survives until the alignment step deletes
    every row, and the resulting error names something entirely unrelated. A
    fetch failure should be reported as a fetch failure, at the point of fetch.
    """
    empty = [name for name in close.columns if close[name].notna().sum() == 0]
    if not empty:
        return
    detail = ", ".join(f"{name} ({tickers[name]})" for name in empty)
    raise RuntimeError(
        f"No data returned for: {detail}.\n"
        f"Most likely transient -- run the script again. If it persists for "
        f"the same ticker, check the symbol is still listed."
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_equity_series(levels: pd.DataFrame) -> pd.Series:
    """Public equity: a blend before ACWI exists, chain-linked to ACWI after.

    ACWI launched in March 2008, which would put the whole financial crisis
    outside the dataset. The blend extends coverage backwards using three funds
    that do span the period, weighted to approximate the same exposure.

    The join is a chain-link, not a concatenation: the blend is rescaled so its
    level meets ACWI at the junction, preserving each segment's own returns.
    Concatenating raw levels would fabricate one enormous return at the seam.
    """
    missing = [k for k in EQUITY_BLEND_TICKERS if k not in levels.columns]
    if missing or "acwi" not in levels.columns:
        raise KeyError(
            f"Equity construction needs {list(EQUITY_BLEND_TICKERS)} and 'acwi'; "
            f"missing {missing + ([] if 'acwi' in levels.columns else ['acwi'])}"
        )

    blend = blend_levels(levels, EQUITY_BLEND_WEIGHTS)
    acwi = levels["acwi"].dropna()

    if acwi.empty:
        raise RuntimeError("ACWI series is empty; cannot build equity")

    spliced = splice_levels(newer=acwi, older=blend)
    spliced.name = "equity"
    return spliced


def fetch_cash_index(
    start: str = DEFAULT_START,
    end: str | None = None,
    ticker: str = CASH_TICKER,
) -> pd.Series:
    """Compound the 13-week T-bill rate into a cash total-return index.

    The quoted figure is an annualised *discount* rate in percent, so 5.25
    means 5.25%. This treats it as a simple annual yield and compounds it on an
    actual/365 basis. That slightly understates a true bond-equivalent yield --
    immaterial for a small cash sleeve, but an approximation, and named here
    rather than buried.
    """
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False
    )
    if raw.empty:
        raise RuntimeError(f"No data returned for the cash rate series ({ticker})")

    rate = raw["Close"]
    if isinstance(rate, pd.DataFrame):
        rate = rate.iloc[:, 0]
    rate = rate.dropna() / 100.0

    if rate.empty:
        raise RuntimeError(f"Cash rate series ({ticker}) is empty after cleaning")

    # Small negative T-bill yields are real, not errors. US bills traded
    # slightly below zero in late 2015 and again in March 2020, when demand for
    # safety was strong enough that buyers accepted a small loss for
    # government paper. Rejecting those would discard genuine history.
    #
    # A large negative is a different matter -- that indicates a bad quote or a
    # misread series, so the guard is set well below anything observed.
    worst = float(rate.min())
    if worst < MIN_PLAUSIBLE_RATE:
        when = rate.idxmin()
        raise RuntimeError(
            f"Cash rate hit {worst:.2%} on {when:%Y-%m-%d}, below the "
            f"{MIN_PLAUSIBLE_RATE:.0%} floor. Small negatives are real (2015, "
            f"March 2020); a figure this low indicates a bad quote or the "
            f"wrong series."
        )

    return compound_rate_to_index(rate)


def compound_rate_to_index(rate: pd.Series) -> pd.Series:
    """Turn a series of annualised rates into a growth index starting at 1.0.

    Separated from the fetch so it can be tested without a network call. Each
    observation's rate is applied over the days until the next observation,
    on an actual/365 basis.
    """
    rate = rate.dropna().sort_index()
    if len(rate) < 2:
        raise ValueError("Need at least 2 rate observations")

    if rate.index.has_duplicates:
        dupes = rate.index[rate.index.duplicated()].unique()
        raise ValueError(
            f"Duplicate dates in the rate series: {list(dupes[:5])}. Each "
            f"would be compounded again over a zero-day gap, which is "
            f"harmless here but indicates the source returned something "
            f"unexpected."
        )

    day_count = rate.index.to_series().diff().dt.days.fillna(0).clip(lower=0)
    growth = (1.0 + rate) ** (day_count / 365.0)
    index = growth.cumprod()
    index.iloc[0] = 1.0
    index.name = "cash"
    return index


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_panel(
    start: str = DEFAULT_START,
    end: str | None = None,
    verbose: bool = True,
) -> PriceHistory:
    """Fetch everything, construct the derived series, align to month-end.

    Returns levels, not returns -- conversion happens downstream, so callers
    that need the price path still have it.
    """
    if verbose:
        print("[market_data] fetching price series...", flush=True)
    levels = fetch_levels(ALL_TICKERS, start=start, end=end)

    if verbose:
        print("[market_data] building equity splice...", flush=True)
    equity = build_equity_series(levels)

    if verbose:
        print("[market_data] fetching cash rate...", flush=True)
    cash = fetch_cash_index(start=start, end=end)

    combined = pd.DataFrame({"equity": equity})
    for name in DIRECT_TICKERS:
        combined[name] = levels[name]
    combined["cash"] = cash

    monthly = PriceHistory(combined.ffill().dropna(how="all")).to_month_end()

    binding, first = monthly.binding_asset()
    if verbose:
        print(
            f"[market_data] common window starts {first:%Y-%m-%d}, "
            f"limited by '{binding}'",
            flush=True,
        )

    return monthly.common_window()


def validate(history: PriceHistory) -> pd.DataFrame:
    """Per-asset data quality report. Read this before trusting any output.

    `zero_return_months` counts months where the price did not move at all --
    almost always a repeated quote rather than a real flat month. A handful is
    normal; many indicates a stale series.

    `min_return` is the splice-error detector: a chain-link done wrong shows up
    as a single catastrophic month.
    """
    levels = history.levels
    rets = history.returns().returns

    return pd.DataFrame(
        {
            "first": levels.apply(lambda s: s.first_valid_index()),
            "last": levels.apply(lambda s: s.last_valid_index()),
            "n_obs": levels.count(),
            "n_missing": levels.isna().sum(),
            "zero_return_months": (rets == 0).sum(),
            "min_return": rets.min(),
            "max_return": rets.max(),
            "ann_return": (1.0 + rets).prod() ** (12 / len(rets)) - 1.0,
            "ann_vol": rets.std(ddof=1) * (12**0.5),
        }
    )
