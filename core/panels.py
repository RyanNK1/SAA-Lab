"""Panel types for price levels and periodic returns.

The separation between `PriceHistory` and `ReturnPanel` is the central design
decision of this project, and it is deliberate.

A covariance or correlation matrix is only meaningful when computed on
*returns*. Computed on price or index *levels*, the result is dominated by
shared trend rather than shared movement: any two series that drift upward over
a long sample will correlate near 1.0 regardless of whether their
period-to-period changes have anything in common.

`PriceHistory` therefore exposes no `cov()` and no `corr()`. The only route
from levels to a second moment runs through `.returns()`. The mistake cannot be
made by accident; it has to be made on purpose, via the clearly labelled
`levels_correlation_bias()` diagnostic at the bottom of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import MONTHS_PER_YEAR


@dataclass(frozen=True)
class PriceHistory:
    """A panel of price or total-return index *levels*, indexed by date.

    Columns are asset names. Deliberately exposes no covariance or correlation
    methods -- see the module docstring.
    """

    levels: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.levels, pd.DataFrame):
            raise TypeError("PriceHistory expects a DataFrame of levels")
        if self.levels.empty:
            raise ValueError("PriceHistory is empty")
        if not isinstance(self.levels.index, pd.DatetimeIndex):
            raise TypeError("PriceHistory index must be a DatetimeIndex")
        if not self.levels.index.is_monotonic_increasing:
            raise ValueError(
                "PriceHistory index must be sorted ascending. Some data "
                "exports return newest-first; sorting silently is worse than "
                "failing, because a reversed index negates every return in the "
                "panel without any other symptom."
            )
        if self.levels.index.has_duplicates:
            dupes = self.levels.index[self.levels.index.duplicated()].unique()
            raise ValueError(f"PriceHistory has duplicate dates: {list(dupes[:5])}")

        non_positive = (self.levels <= 0).any()
        if bool(non_positive.any()):
            bad = list(non_positive[non_positive].index)
            raise ValueError(
                f"Non-positive levels in {bad}. Simple returns would be "
                f"undefined or imply a negative price."
            )

    @property
    def assets(self) -> list[str]:
        return list(self.levels.columns)

    def __len__(self) -> int:
        return len(self.levels)

    def to_month_end(
        self, drop_incomplete: bool = True, tolerance_days: int = 5
    ) -> PriceHistory:
        """Resample to month-end observations, taking the last level in each.

        By default the final month is dropped when the data does not actually
        reach the end of it. Resampling labels every bucket with its month-end
        date, so a run made on the 11th produces a row dated the 31st holding
        eleven days of movement. That row is a partial-month return wearing a
        full month's label, and every downstream calculation would treat it as
        a complete observation -- understating that month's return and, through
        it, the annualised figures.

        `tolerance_days` allows for the month ending on a weekend or holiday,
        when the last trading day legitimately falls a few days short. Only the
        final bucket is checked; earlier gaps are the data source's business,
        not a partial-period artefact.
        """
        resampled = self.levels.resample("ME").last().dropna(how="all")

        if drop_incomplete and len(resampled) > 0:
            last_observation = self.levels.index[-1]
            last_bucket = resampled.index[-1]
            shortfall = (last_bucket - last_observation).days
            if shortfall > tolerance_days:
                resampled = resampled.iloc[:-1]
                if len(resampled) == 0:
                    raise ValueError(
                        f"Dropping the incomplete final month left nothing. "
                        f"Data ends {last_observation:%Y-%m-%d}, short of the "
                        f"{last_bucket:%Y-%m-%d} month end."
                    )

        return PriceHistory(resampled)

    def returns(self, periods_per_year: int = MONTHS_PER_YEAR) -> ReturnPanel:
        """Simple period-over-period returns. Drops the first, undefined row."""
        rets = self.levels.pct_change().iloc[1:]
        return ReturnPanel(rets, periods_per_year=periods_per_year)

    def first_valid(self) -> pd.Series:
        """First date each asset has data. NaT where an asset is entirely empty."""
        return self.levels.apply(lambda s: s.first_valid_index())

    def binding_asset(self) -> tuple[str, pd.Timestamp]:
        """Which asset's start date limits the common window, and when.

        Call this before `common_window()` so the truncation is a reported
        fact rather than a silent one.
        """
        starts = self.first_valid()
        if starts.isna().any():
            empty = list(starts[starts.isna()].index)
            raise ValueError(f"These assets have no data at all: {empty}")
        return str(starts.idxmax()), pd.Timestamp(starts.max())

    def common_window(self) -> PriceHistory:
        """Truncate to the span where every asset has data.

        Raises with a useful diagnosis rather than returning an empty panel,
        because the intersection of anything with an empty column is empty and
        the resulting error would otherwise surface far from its cause.
        """
        trimmed = self.levels.dropna(how="any")
        if len(trimmed) >= 2:
            return PriceHistory(trimmed)

        empty = [c for c in self.levels.columns if self.levels[c].notna().sum() == 0]
        if empty:
            raise ValueError(
                f"Common window is empty because these assets have no data at "
                f"all: {empty}. Fix the fetch before going further."
            )
        coverage = {
            c: (
                f"{self.levels[c].first_valid_index():%Y-%m-%d}"
                f" to {self.levels[c].last_valid_index():%Y-%m-%d}"
            )
            for c in self.levels.columns
        }
        raise ValueError(
            f"Assets share only {len(trimmed)} dates. Coverage: {coverage}"
        )


@dataclass(frozen=True)
class ReturnPanel:
    """A panel of periodic *simple* returns. The only source of second moments."""

    returns: pd.DataFrame
    periods_per_year: int = MONTHS_PER_YEAR

    def __post_init__(self) -> None:
        if not isinstance(self.returns, pd.DataFrame):
            raise TypeError("ReturnPanel expects a DataFrame of returns")
        if self.returns.isna().any().any():
            bad = list(self.returns.columns[self.returns.isna().any()])
            raise ValueError(
                f"NaNs in {bad}. Align and drop before constructing -- silent "
                f"pairwise deletion estimates each covariance entry on a "
                f"different sample, which can produce a matrix that is not "
                f"positive semi-definite."
            )
        if len(self.returns) < 2:
            raise ValueError("ReturnPanel needs at least 2 observations")
        if self.periods_per_year < 1:
            raise ValueError("periods_per_year must be positive")

        if (self.returns <= -1.0).any().any():
            bad = list(self.returns.columns[(self.returns <= -1.0).any()])
            raise ValueError(
                f"Returns of -100% or worse in {bad}. A simple return cannot be "
                f"-1 or below unless the asset went to zero; this usually means "
                f"two price series were concatenated without chain-linking."
            )

    @property
    def assets(self) -> list[str]:
        return list(self.returns.columns)

    def __len__(self) -> int:
        return len(self.returns)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.returns.index[0])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.returns.index[-1])

    def select(self, assets: list[str]) -> ReturnPanel:
        """Restrict to a subset of assets -- the user's asset-class picker."""
        missing = [a for a in assets if a not in self.returns.columns]
        if missing:
            raise KeyError(f"Unknown assets {missing}. Have: {self.assets}")
        return ReturnPanel(self.returns[assets], self.periods_per_year)

    def between(self, start, end) -> ReturnPanel:
        """Restrict to a date range -- the user's period picker.

        Bounds are inclusive. Out-of-range bounds are allowed and simply clip;
        an empty result raises rather than returning a panel nothing can use.
        """
        sliced = self.returns.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if len(sliced) < 2:
            raise ValueError(
                f"{start} to {end} contains {len(sliced)} observations. "
                f"Panel covers {self.start:%Y-%m} to {self.end:%Y-%m}."
            )
        return ReturnPanel(sliced, self.periods_per_year)

    # -- second moments: only available here, never on PriceHistory --------

    def cov(self) -> pd.DataFrame:
        """Sample covariance of periodic returns (not annualised)."""
        return self.returns.cov()

    def corr(self) -> pd.DataFrame:
        """Sample correlation of periodic returns."""
        return self.returns.corr()

    def ann_cov(self) -> pd.DataFrame:
        return self.cov() * self.periods_per_year

    # -- first moments ------------------------------------------------------

    def ann_return(self) -> pd.Series:
        """Geometric annualised return (CAGR) per asset.

        Geometric, not arithmetic-annualised. Compounding the arithmetic mean
        overstates realised growth by roughly half the variance -- about 130bp
        a year on a 16% volatility asset, which is large enough to change
        conclusions.
        """
        n = len(self.returns)
        growth = (1.0 + self.returns).prod()
        return growth ** (self.periods_per_year / n) - 1.0

    def arith_ann_return(self) -> pd.Series:
        """Arithmetic mean compounded to a year. For comparison only."""
        return (1.0 + self.returns.mean()) ** self.periods_per_year - 1.0

    def ann_vol(self) -> pd.Series:
        """Annualised standard deviation of periodic returns (sample, ddof=1)."""
        return self.returns.std(ddof=1) * np.sqrt(self.periods_per_year)

    def cumulative(self) -> pd.DataFrame:
        """Growth of 1 unit per asset. The equity curve, before weighting."""
        return (1.0 + self.returns).cumprod()


# ---------------------------------------------------------------------------
# Chain-linking
# ---------------------------------------------------------------------------

def splice_levels(newer: pd.Series, older: pd.Series) -> pd.Series:
    """Chain-link two level series into one continuous history.

    Long asset-class histories usually require this. ACWI does not exist before
    March 2008, so a blend of longer-running funds is chained behind it.

    `older` is rescaled so its level meets `newer` at the junction. Each
    segment's own *returns* are preserved exactly; only the arbitrary level is
    adjusted.

    Concatenating raw levels instead fabricates one enormous return at the
    seam -- a series at 151 followed by one at 1.00 implies a -99.3% month.
    That single wrong observation is easy to miss and contaminates every
    statistic computed afterwards.

    If the two series overlap, the rescale anchors on the earliest shared date.
    Without overlap the junction month's return is unobservable and is
    implicitly zero, which is why overlap is preferred.
    """
    newer = newer.dropna().sort_index()
    older = older.dropna().sort_index()

    if newer.empty or older.empty:
        raise ValueError("splice_levels needs two non-empty series")
    if older.index[0] >= newer.index[0]:
        raise ValueError(
            f"`older` must start before `newer`: older starts "
            f"{older.index[0]:%Y-%m-%d}, newer starts {newer.index[0]:%Y-%m-%d}"
        )

    overlap = older.index.intersection(newer.index)
    if len(overlap) > 0:
        anchor = overlap[0]
        scale = newer.loc[anchor] / older.loc[anchor]
    else:
        scale = newer.iloc[0] / older.iloc[-1]

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Splice scale factor is invalid ({scale})")

    rescaled = older[older.index < newer.index[0]] * scale
    out = pd.concat([rescaled, newer]).sort_index()
    out.name = newer.name
    return out


def blend_levels(
    components: pd.DataFrame, weights: dict[str, float], tol: float = 1e-8
) -> pd.Series:
    """Combine several level series into one, rebalanced every period.

    Used to approximate a broad index from its parts before the index itself
    exists. The blend is built from *returns* -- a weighted average of levels
    would be meaningless, since the level of each component is an arbitrary
    scale.

    Rebalancing every period is what makes the weighted sum of returns the
    blend's return, and keeps the composition matching the stated weights
    rather than drifting toward whichever component grew fastest.
    """
    missing = set(weights) - set(components.columns)
    if missing:
        raise KeyError(f"No column for {sorted(missing)}")

    total = sum(weights.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Blend weights must sum to 1.0, got {total:.10f}")

    frame = components[list(weights)].dropna(how="any")
    if len(frame) < 2:
        raise ValueError(
            f"Blend components share only {len(frame)} dates; need at least 2"
        )

    rets = frame.pct_change().iloc[1:]
    w = pd.Series(weights, dtype=float).reindex(rets.columns)
    blended = rets.mul(w, axis=1).sum(axis=1)

    level = (1.0 + blended).cumprod()
    first = pd.Series([1.0], index=[frame.index[0]])
    return pd.concat([first, level]).sort_index()


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def levels_correlation_bias(
    history: PriceHistory, periods_per_year: int = MONTHS_PER_YEAR
) -> pd.DataFrame:
    """Correlation computed on levels versus on returns, side by side.

    The one function permitted to correlate levels, and it exists only to
    demonstrate why nothing else should. The `corr_returns` column is the
    meaningful figure; `corr_levels` is the artefact.

    Returns one row per asset pair, sorted by the size of the discrepancy.
    """
    levels_corr = history.levels.corr()
    returns_corr = history.returns(periods_per_year).corr()

    rows = []
    assets = history.assets
    for i, a in enumerate(assets):
        for b in assets[i + 1 :]:
            rows.append(
                {
                    "asset_a": a,
                    "asset_b": b,
                    "corr_levels": levels_corr.loc[a, b],
                    "corr_returns": returns_corr.loc[a, b],
                    "overstatement": levels_corr.loc[a, b] - returns_corr.loc[a, b],
                }
            )
    return pd.DataFrame(rows).sort_values(
        "overstatement", key=abs, ascending=False, ignore_index=True
    )
