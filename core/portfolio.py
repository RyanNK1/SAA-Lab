"""Measurement of a portfolio: every number the platform reports.

Given a weight vector and a panel of returns, produce return, volatility,
Sharpe, Sortino, drawdown, recovery, and risk contributions. No optimisation
happens here -- Step 4 searches over weights, and this is the thing it
searches with, so every objective the optimizer supports must be computable
here first.

Two families of statistic, and the distinction matters:

**Distribution statistics** (return, volatility, Sharpe, Sortino) describe the
spread of monthly returns. Shuffle the months into any order and none of them
move.

**Path statistics** (drawdown, recovery, time underwater) read the sequence.
Twelve scattered bad months and twelve consecutive ones give identical Sharpe
ratios and completely different experiences -- one is an annoyance, the other
is a portfolio down 46% with the holder very likely to have sold at the
bottom. A spreadsheet cannot ask this question; it has distributions but no
path.

Everything here assumes the portfolio is rebalanced every period, which is
what makes the weighted sum of asset returns the portfolio return. Step 3
generalises to annual, threshold and never, and supplies a path for these
functions to measure rather than reimplementing the statistics per mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.panels import ReturnPanel

CASH = "cash"

# Below this annualised volatility, ratios stop being meaningful: a portfolio
# sitting in cash has a tiny excess return over a tinier denominator and can
# score arbitrarily high. Reporting a Sharpe of 30 would be arithmetically
# correct and financially worthless.
MIN_MEANINGFUL_VOL = 0.001


def _ratio(numerator: float, denominator: float) -> float:
    """A risk-adjusted ratio, handling a vanishing denominator by direction.

    When the denominator is effectively zero the ratio is undefined, and which
    answer is right depends on the numerator's sign:

      positive excess -> unbounded, reported as +inf. The portfolio earned
        more than cash without ever falling short of it.
      negative excess -> -inf. It underperformed cash with no compensating
        movement to explain it.
      no excess -> 0.0. Nothing happened either way.

    Returning 0.0 in every case, as a single guard would, is actively
    misleading: it says "worst possible" about a portfolio that never once
    underperformed, and an optimizer maximising the ratio would then avoid
    exactly the portfolio it should prefer.
    """
    if denominator >= MIN_MEANINGFUL_VOL:
        return numerator / denominator
    if abs(numerator) < 1e-12:
        return 0.0
    return float("inf") if numerator > 0 else float("-inf")


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def as_weights(
    weights: pd.Series | dict[str, float], assets: list[str], tol: float = 1e-8
) -> pd.Series:
    """Validate a weight vector against a set of assets.

    Errors name the offending asset. A weight vector that silently drops an
    asset, or silently includes one the panel does not have, produces a
    plausible number for the wrong portfolio.
    """
    w = pd.Series(weights, dtype=float)

    if w.isna().any():
        raise ValueError(f"Missing weight values for {list(w[w.isna()].index)}")

    missing = set(assets) - set(w.index)
    extra = set(w.index) - set(assets)
    if missing:
        raise KeyError(f"No weight supplied for: {sorted(missing)}")
    if extra:
        raise KeyError(f"Weight supplied for unknown asset: {sorted(extra)}")

    if (w < -tol).any():
        raise ValueError(
            f"Negative weights for {list(w[w < -tol].index)}. Short positions "
            f"are outside this tool's scope."
        )

    total = float(w.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.0, got {total:.10f}")

    return w.reindex(assets)


def portfolio_returns(
    panel: ReturnPanel, weights: pd.Series | dict[str, float]
) -> pd.Series:
    """The realised return series of a portfolio rebalanced every period."""
    w = as_weights(weights, panel.assets)
    return panel.returns.mul(w, axis=1).sum(axis=1)


def cash_returns(panel: ReturnPanel) -> pd.Series | None:
    """The contemporaneous cash return series, if the panel has one."""
    if CASH not in panel.assets:
        return None
    return panel.returns[CASH]


# ---------------------------------------------------------------------------
# Path statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Drawdown:
    """The worst peak-to-trough decline, and how long it took to undo."""

    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    months_to_trough: int
    months_to_recover: int | None
    months_underwater: int

    @property
    def recovered(self) -> bool:
        return self.recovery_date is not None

    def describe(self) -> str:
        if self.recovered:
            tail = f"recovered by {self.recovery_date:%Y-%m} ({self.months_to_recover} months)"
        else:
            tail = "never recovered within the period"
        return (
            f"{self.max_drawdown:.1%} from {self.peak_date:%Y-%m} to "
            f"{self.trough_date:%Y-%m}, {tail}"
        )


def _wealth_curve(returns: pd.Series) -> pd.Series:
    """Growth of 1 unit, including the starting point before any returns.

    The leading 1.0 is not cosmetic. Without it the running maximum begins at
    the first period's *closing* value, so a portfolio that falls from the very
    first month measures its drawdown from an already-depressed level. A
    period beginning at a market peak -- October 2007, say -- would report a
    materially understated worst case, which is precisely the case where the
    number matters most.
    """
    curve = (1.0 + returns).cumprod()
    if len(returns) == 0:
        return curve
    step = returns.index[1] - returns.index[0] if len(returns) > 1 else pd.Timedelta(days=1)
    origin = pd.Series([1.0], index=[returns.index[0] - step])
    return pd.concat([origin, curve])


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Decline from the running peak, at every point. Zero or negative.

    Indexed on the return dates; the implicit starting point is used to seed
    the running maximum but is not reported, since no decline has happened yet.
    """
    curve = _wealth_curve(returns)
    drawdowns = curve / curve.cummax() - 1.0
    return drawdowns.iloc[1:]


def analyse_drawdown(returns: pd.Series) -> Drawdown:
    """Locate the worst decline and measure the round trip.

    Recovery is defined as returning to the *prior peak*, not to some arbitrary
    level. That is what an investor experiences: the portfolio is not whole
    until it has undone the loss.

    `months_underwater` counts every month spent below a previous peak across
    the whole period, not just this episode -- a portfolio that spends fifteen
    years below a high-water mark is telling you something a single worst-case
    number does not.
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 observations to measure drawdown")

    curve = _wealth_curve(returns)
    peaks = curve.cummax()
    drawdowns = curve / peaks - 1.0

    trough_date = drawdowns.iloc[1:].idxmin()
    max_dd = float(drawdowns.loc[trough_date])

    # The peak is the last date at or before the trough where the curve was at
    # its running maximum. This may be the implicit starting point, when the
    # portfolio declined from the outset.
    before_trough = curve.loc[:trough_date]
    peak_date = before_trough.idxmax()
    peak_value = float(before_trough.loc[peak_date])

    after_trough = curve.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None

    index = curve.index
    months_to_trough = int(index.get_loc(trough_date) - index.get_loc(peak_date))
    months_to_recover = (
        int(index.get_loc(recovery_date) - index.get_loc(trough_date))
        if recovery_date is not None
        else None
    )

    return Drawdown(
        max_drawdown=max_dd,
        peak_date=pd.Timestamp(peak_date),
        trough_date=pd.Timestamp(trough_date),
        recovery_date=pd.Timestamp(recovery_date) if recovery_date is not None else None,
        months_to_trough=months_to_trough,
        months_to_recover=months_to_recover,
        months_underwater=int((drawdowns.iloc[1:] < 0).sum()),
    )


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioStats:
    """Everything measurable about one allocation over one period."""

    weights: pd.Series
    returns: pd.Series
    periods_per_year: int

    model_return: float
    realised_return: float
    volatility: float
    excess_return: float
    downside_deviation: float
    drawdown: Drawdown

    # Note on the two return measures reported here:
    #
    #   `realised_return` is geometric (CAGR) -- what the money actually did.
    #   `excess_return` is the arithmetic mean of monthly excess returns,
    #   annualised, which is the standard Sharpe numerator.
    #
    # These are deliberately different, not an inconsistency. The Sharpe ratio
    # is defined on the arithmetic mean because that is what pairs with a
    # standard deviation computed from the same observations; using a
    # geometric numerator over an arithmetic denominator mixes two things.
    # `realised_return` is the honest answer to "what did I earn", and
    # `excess_return` is the honest numerator for a risk-adjusted ratio. Both
    # are reported so neither has to stand in for the other.

    @property
    def n_periods(self) -> int:
        return len(self.returns)

    @property
    def return_gap(self) -> float:
        """Realised minus model return.

        The model return is the weighted average of per-asset returns -- what a
        spreadsheet computes. The realised return compounds the actual monthly
        path. They differ, mostly because rebalancing across imperfectly
        correlated assets harvests a little volatility. Reporting both prevents
        an apples-to-oranges comparison creeping in once the optimizer exists.
        """
        return self.realised_return - self.model_return

    @property
    def sharpe(self) -> float:
        """Excess return over cash, per unit of total volatility."""
        return _ratio(self.excess_return, self.volatility)

    @property
    def sortino(self) -> float:
        """Excess return over cash, per unit of *downside* volatility.

        Same numerator as Sharpe, different denominator. Only months below the
        target count toward the denominator, so a portfolio is not penalised
        for moving sharply upward.

        A portfolio that never falls below the target has zero downside
        deviation and an unbounded Sortino. That is reported as infinity, not
        as zero -- zero would say "worst possible" about a portfolio that never
        once underperformed cash, and any optimizer maximising the ratio would
        then avoid precisely the portfolio it should prefer.
        """
        return _ratio(self.excess_return, self.downside_deviation)

    @property
    def max_drawdown(self) -> float:
        return self.drawdown.max_drawdown

    def to_dict(self) -> dict[str, float]:
        return {
            "model_return": self.model_return,
            "realised_return": self.realised_return,
            "return_gap": self.return_gap,
            "volatility": self.volatility,
            "excess_return": self.excess_return,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "months_to_recover": self.drawdown.months_to_recover,
            "months_underwater": self.drawdown.months_underwater,
        }


def downside_deviation(
    returns: pd.Series, target: pd.Series | float, periods_per_year: int
) -> float:
    """Annualised deviation of returns below a target.

    The target is the risk-free rate, not zero. Using zero would treat "earned
    less than cash" as an acceptable outcome, which it is not -- an investor
    could have held cash instead with no risk at all.

    Shortfalls are squared and averaged over *all* periods, not only the ones
    below target. Dividing by the count of bad months instead would make a
    portfolio with rare-but-severe losses look better than one with frequent
    mild ones, which inverts the thing being measured.
    """
    shortfall = np.minimum(returns - target, 0.0)
    return float(np.sqrt((shortfall**2).mean()) * np.sqrt(periods_per_year))


def portfolio_stats(
    panel: ReturnPanel,
    weights: pd.Series | dict[str, float],
    risk_free: pd.Series | float | None = None,
) -> PortfolioStats:
    """Measure one allocation over the panel's period.

    `risk_free` may be a series (the contemporaneous cash return, preferred), a
    scalar annual rate, or None. When None, the panel's own cash column is used
    if present, otherwise zero.

    A single fixed rate across a long window is a trap: cash paid near zero
    until 2022 and around 5% afterwards, so a fixed early-sample rate would
    credit a cash-heavy portfolio with an excess return it never earned. The
    series form is used wherever available.
    """
    w = as_weights(weights, panel.assets)
    rets = portfolio_returns(panel, w)
    n = len(rets)
    ppy = panel.periods_per_year

    if risk_free is None:
        rf_series = cash_returns(panel)
        rf = rf_series if rf_series is not None else 0.0
    elif isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(rets.index)
        if rf.isna().any():
            raise ValueError("risk_free series does not cover the panel's dates")
    else:
        rf = (1.0 + float(risk_free)) ** (1.0 / ppy) - 1.0

    excess = rets - rf

    realised = float((1.0 + rets).prod() ** (ppy / n) - 1.0)
    model = float(w @ panel.ann_return())
    vol = float(rets.std(ddof=1) * np.sqrt(ppy))

    return PortfolioStats(
        weights=w,
        returns=rets,
        periods_per_year=ppy,
        model_return=model,
        realised_return=realised,
        volatility=vol,
        excess_return=float(excess.mean() * ppy),
        downside_deviation=downside_deviation(rets, rf, ppy),
        drawdown=analyse_drawdown(rets),
    )


# ---------------------------------------------------------------------------
# Risk attribution
# ---------------------------------------------------------------------------

def risk_contributions(
    panel: ReturnPanel, weights: pd.Series | dict[str, float]
) -> pd.DataFrame:
    """How much of the portfolio's risk each asset actually carries.

    Weight is not risk. A 10% holding in something volatile and correlated with
    everything else can carry 30% of portfolio movement, while a 20% holding in
    something uncorrelated carries far less than 20%.

    This is the table that exposes a label doing the talking. The private
    equity proxy correlates around 0.9 with public equity, so a portfolio that
    looks diversified across five buckets can be one bet in three costumes.

    Percentages sum to 1 by Euler's theorem: volatility is homogeneous of
    degree one in the weights, so the weighted marginal contributions add back
    to total volatility exactly. That identity is a free correctness check.
    """
    w = as_weights(weights, panel.assets)
    ann_cov = panel.ann_cov()
    wv = w.to_numpy()

    variance = float(wv @ ann_cov.to_numpy() @ wv)
    if variance < 0:
        raise ValueError(
            f"Negative portfolio variance ({variance:.3e}); the covariance "
            f"matrix is not positive semi-definite"
        )

    vol = float(np.sqrt(variance))
    if vol < MIN_MEANINGFUL_VOL:
        marginal = np.zeros_like(wv)
        contribution = np.zeros_like(wv)
        pct = np.where(wv > 0, wv, 0.0)
        pct = pct / pct.sum() if pct.sum() > 0 else pct
    else:
        marginal = ann_cov.to_numpy() @ wv / vol
        contribution = wv * marginal
        pct = contribution / vol

    return pd.DataFrame(
        {
            "weight": w,
            "ann_vol": panel.ann_vol(),
            "marginal_risk": marginal,
            "risk_contribution": contribution,
            "pct_of_risk": pct,
        },
        index=panel.assets,
    )
