"""Finding the best allocation for a period.

Four objectives, solved two different ways, because two of them are smooth
functions of the weights and two are not.

**Smooth.** Volatility comes straight from the covariance matrix as `w' S w`.
Nudge a weight slightly and the answer moves slightly, so a gradient solver can
feel which direction is downhill and walk there. Sharpe is smooth for the same
reason. Both are solved exactly with SLSQP.

**Not smooth.** Sortino counts only the months that fell below a target; nudge
a weight and a month sitting just above the line drops below it, so the
objective jumps rather than slides. Drawdown depends on *which* months were
worst and in what order, so a small weight change can relocate the worst
decline from one crisis to another entirely. Neither has a usable slope. A
gradient solver would stop at the first flat spot it found and report it with
full confidence.

Those two are solved by sampling: measure a large number of feasible
allocations, keep the best, refine locally around it. That is a very strong
candidate rather than a proven optimum, and every result carries a `method`
field saying which it is. Presenting all four with equal confidence would be
dishonest.

The sampling pass is shared. Generating the allocations is the expensive part;
measuring all four objectives on each is nearly free, so one pass answers both
sampled objectives, cross-checks the two exact ones, and supplies the
near-optimal range without a second search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.config import Objective
from core.panels import ReturnPanel
from core.portfolio import MIN_MEANINGFUL_VOL, PortfolioStats, portfolio_stats

DEFAULT_SAMPLES = 20_000
DEFAULT_TOLERANCE = 0.02
_MAX_ITER = 500
_FTOL = 1e-12


class Method(str, Enum):
    """How an answer was reached. Reported so the two are not confused."""

    EXACT = "exact"
    SAMPLED = "sampled"


# Whether each objective is maximised, and whether it can be solved exactly.
_MAXIMISE: dict[Objective, bool] = {
    Objective.MAX_SHARPE: True,
    Objective.MAX_SORTINO: True,
    Objective.MIN_VOLATILITY: False,
    Objective.MIN_DRAWDOWN: False,
}

_EXACTLY_SOLVABLE = {Objective.MIN_VOLATILITY, Objective.MAX_SHARPE}


def score(stats: PortfolioStats, objective: Objective) -> float:
    """The value being optimised, for one measured allocation."""
    if objective is Objective.MAX_SHARPE:
        return stats.sharpe
    if objective is Objective.MAX_SORTINO:
        return stats.sortino
    if objective is Objective.MIN_VOLATILITY:
        return stats.volatility
    if objective is Objective.MIN_DRAWDOWN:
        return stats.max_drawdown
    raise ValueError(f"Unknown objective {objective}")


def is_better(candidate: float, incumbent: float, objective: Objective) -> bool:
    """Direction-aware comparison. Drawdown is negative, so larger is better."""
    if not np.isfinite(candidate):
        return False
    if _MAXIMISE[objective] or objective is Objective.MIN_DRAWDOWN:
        return candidate > incumbent
    return candidate < incumbent


@dataclass(frozen=True)
class OptimizationResult:
    """One solved allocation, with the method that produced it."""

    weights: pd.Series
    stats: PortfolioStats
    objective: Objective
    method: Method
    near_optimal: pd.DataFrame = field(default_factory=pd.DataFrame)
    tolerance: float = DEFAULT_TOLERANCE
    n_samples: int = 0

    @property
    def value(self) -> float:
        return score(self.stats, self.objective)

    def ranges(self) -> pd.DataFrame:
        """The span each asset's weight takes across near-optimal allocations.

        Reporting a single allocation implies a precision the data does not
        support: structurally different portfolios routinely land within a
        fraction of a percent of each other. This is the honest answer --
        "equity anywhere between 45% and 60%, and it barely matters".

        The best allocation is folded into the span. For an exactly-solved
        objective the optimum comes from the solver while the near-optimal set
        comes from sampling, so the optimum is not in that set and the raw
        sampled range can sit entirely to one side of it -- producing a "range"
        that excludes the answer it describes. Widening to include the best
        allocation keeps the two halves of the output talking about the same
        thing.
        """
        if self.near_optimal.empty:
            return pd.DataFrame(
                {
                    "best": self.weights,
                    "low": self.weights,
                    "high": self.weights,
                    "spread": 0.0,
                }
            )

        assets = list(self.weights.index)
        low = np.minimum(self.near_optimal[assets].min(), self.weights)
        high = np.maximum(self.near_optimal[assets].max(), self.weights)

        return pd.DataFrame(
            {"best": self.weights, "low": low, "high": high, "spread": high - low}
        )

    def describe(self) -> str:
        confidence = (
            "exact" if self.method is Method.EXACT else f"best of {self.n_samples:,}"
        )
        return f"{self.objective.value}: {self.value:.4f} ({confidence})"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_allocations(
    n_assets: int,
    n_samples: int = DEFAULT_SAMPLES,
    max_weight: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Feasible long-only allocations spread across the simplex.

    Dirichlet draws cover the interior. A share of concentrated draws is mixed
    in because pure Dirichlet sampling rarely lands near a corner, and several
    of these objectives have their optimum there -- minimum drawdown in
    particular wants to sit almost entirely in cash.
    """
    if n_assets < 1:
        raise ValueError("Need at least one asset")
    if n_samples < 1:
        raise ValueError("Need at least one sample")
    if not 1.0 / n_assets <= max_weight <= 1.0:
        raise ValueError(
            f"max_weight must be between 1/n ({1.0 / n_assets:.4f}) and 1.0; "
            f"below 1/n no long-only allocation can sum to one"
        )

    rng = np.random.default_rng(seed)

    n_corner = max(n_assets, n_samples // 10)
    n_spread = n_samples - n_corner - 1

    draws = [np.full((1, n_assets), 1.0 / n_assets)]  # equal weight, always
    if n_spread > 0:
        draws.append(rng.dirichlet(np.ones(n_assets), size=n_spread))
    if n_corner > 0:
        # Low concentration parameter pushes mass toward a few assets.
        draws.append(rng.dirichlet(np.full(n_assets, 0.15), size=n_corner))

    samples = np.vstack(draws)

    if max_weight < 1.0:
        samples = _apply_cap(samples, max_weight)

    return samples


def _apply_cap(samples: np.ndarray, max_weight: float) -> np.ndarray:
    """Bring every allocation under a per-asset cap while still summing to one.

    Clipping and renormalising does not work: renormalising scales every weight
    up, which pushes assets back over the cap, and iterating that never
    converges. This redistributes the shortfall only into assets that still
    have headroom, so a capped asset stays capped.
    """
    capped = np.minimum(samples, max_weight)

    for _ in range(64):
        deficit = 1.0 - capped.sum(axis=1)
        if np.abs(deficit).max() < 1e-12:
            break

        headroom = max_weight - capped
        available = headroom.sum(axis=1)

        # Where nothing can absorb the shortfall the cap is infeasible, which
        # the caller already ruled out by validating max_weight >= 1/n.
        share = np.divide(
            headroom,
            available[:, None],
            out=np.zeros_like(headroom),
            where=available[:, None] > 1e-15,
        )
        capped = np.minimum(capped + share * deficit[:, None], max_weight)

    return capped


def _measure_samples(
    panel: ReturnPanel,
    samples: np.ndarray,
    risk_free: pd.Series | float | None,
) -> pd.DataFrame:
    """Measure every objective on every sampled allocation.

    Vectorised where possible: the return series for all samples is one matrix
    product, and volatility comes from the covariance matrix without touching
    the path. Only the path statistics need per-sample work.
    """
    assets = panel.assets
    rets = panel.returns.to_numpy()
    ppy = panel.periods_per_year

    if risk_free is None:
        rf = panel.returns["cash"].to_numpy() if "cash" in assets else 0.0
    elif isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(panel.returns.index).to_numpy()
    else:
        rf = (1.0 + float(risk_free)) ** (1.0 / ppy) - 1.0

    # (n_periods, n_samples): every sampled portfolio's return series at once.
    paths = rets @ samples.T
    excess = paths - np.asarray(rf).reshape(-1, 1)

    n = paths.shape[0]
    growth = np.prod(1.0 + paths, axis=0)
    realised = np.where(growth > 0, np.abs(growth) ** (ppy / n) - 1.0, -1.0)
    vol = paths.std(axis=0, ddof=1) * np.sqrt(ppy)
    excess_ann = excess.mean(axis=0) * ppy

    shortfall = np.minimum(excess, 0.0)
    downside = np.sqrt((shortfall**2).mean(axis=0)) * np.sqrt(ppy)

    curve = np.cumprod(1.0 + paths, axis=0)
    curve = np.vstack([np.ones((1, curve.shape[1])), curve])
    running_peak = np.maximum.accumulate(curve, axis=0)
    drawdowns = curve / running_peak - 1.0
    max_dd = drawdowns.min(axis=0)

    # Months spent below a previous high-water mark, over the whole period.
    # The leading row is the starting point, which cannot be underwater.
    underwater = (drawdowns[1:] < -1e-12).sum(axis=0)

    # Months from the worst trough back to the peak that preceded it.
    recovery, recovered = _months_to_recover(curve, drawdowns)

    # Must match `portfolio._ratio` exactly. A vectorised shortcut that scores
    # the degenerate case differently from the measurement layer means the
    # search is optimising something the reported statistics disagree with --
    # and the disagreement is silent.
    sharpe = _vector_ratio(excess_ann, vol)
    sortino = _vector_ratio(excess_ann, downside)

    frame = pd.DataFrame(samples, columns=assets)
    frame["months_underwater"] = underwater
    # `recovered` is deliberately not stored as a separate boolean column. A
    # bool alongside floats makes every extracted row object-dtype, so pulling
    # the weights out of a result stops behaving like numbers. `np.isinf` on
    # months_to_recover answers the same question without that cost.
    frame["months_to_recover"] = recovery
    frame["realised_return"] = realised
    frame["volatility"] = vol
    frame["sharpe"] = sharpe
    frame["sortino"] = sortino
    frame["max_drawdown"] = max_dd
    return frame


def _months_to_recover(
    curve: np.ndarray, drawdowns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Periods from each portfolio's worst trough back to its prior peak.

    Two allocations with the same worst drawdown are very different
    propositions if one is whole again in eight months and the other takes
    forty. That difference lives entirely in the path.

    Where the peak was never regained, infinity is returned rather than the
    periods remaining. Returning the remainder would be a lower bound, and a
    plausible-looking one -- but it would rank a portfolio that never recovered
    from a late trough *ahead* of one that genuinely recovered in five months.
    Infinity sorts last, which is the honest ordering, and the companion
    boolean says which figures are real measurements.
    """
    n_periods, n_samples = curve.shape
    trough_index = drawdowns.argmin(axis=0)
    recovery = np.empty(n_samples, dtype=float)
    recovered = np.zeros(n_samples, dtype=bool)

    for j in range(n_samples):
        trough = trough_index[j]
        peak_value = curve[: trough + 1, j].max()
        after = curve[trough:, j]
        regained = np.flatnonzero(after >= peak_value)
        if regained.size:
            recovery[j] = float(regained[0])
            recovered[j] = True
        else:
            recovery[j] = np.inf

    return recovery, recovered


def _vector_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Vectorised form of `portfolio._ratio`, kept deliberately identical.

    A near-zero denominator makes the ratio unbounded, and the sign of the
    numerator decides which infinity. Collapsing all of those to zero, as a
    single guard would, tells the search that a portfolio which never fell
    short of cash is the worst available -- the exact inversion the scalar
    version exists to avoid.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator >= MIN_MEANINGFUL_VOL,
        )

    degenerate = denominator < MIN_MEANINGFUL_VOL
    ratio = np.where(
        degenerate & (numerator > 1e-12), np.inf, ratio
    )
    ratio = np.where(degenerate & (numerator < -1e-12), -np.inf, ratio)
    return ratio


_COLUMN_FOR: dict[Objective, str] = {
    Objective.MAX_SHARPE: "sharpe",
    Objective.MAX_SORTINO: "sortino",
    Objective.MIN_VOLATILITY: "volatility",
    Objective.MIN_DRAWDOWN: "max_drawdown",
}


def _near_optimal(
    measured: pd.DataFrame, objective: Objective, best: float, tolerance: float
) -> pd.DataFrame:
    """Every sampled allocation within `tolerance` of the best value.

    Tolerance is relative to the magnitude of the best value, so it means the
    same thing for a Sharpe of 0.5 and a drawdown of -40%.
    """
    column = _COLUMN_FOR[objective]
    if not np.isfinite(best):
        return measured.iloc[0:0].copy()

    band = abs(best) * tolerance
    measured = measured[np.isfinite(measured[column])]

    if _MAXIMISE[objective] or objective is Objective.MIN_DRAWDOWN:
        keep = measured[column] >= best - band
    else:
        keep = measured[column] <= best + band

    return measured[keep].copy()


# ---------------------------------------------------------------------------
# Exact solvers
# ---------------------------------------------------------------------------

def _budget_constraint() -> dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _solve_smooth(
    objective_fn: Callable[[np.ndarray], float],
    n_assets: int,
    max_weight: float,
    starts: list[np.ndarray],
) -> np.ndarray:
    """SLSQP from several starting points, keeping the best feasible answer."""
    bounds = [(0.0, max_weight)] * n_assets
    constraints = [_budget_constraint()]

    best_w, best_f = None, np.inf
    message = "no start converged"

    for x0 in starts:
        result = minimize(
            objective_fn,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": _MAX_ITER, "ftol": _FTOL},
        )
        if result.success and result.fun < best_f:
            best_w, best_f = result.x, result.fun
            message = result.message

    if best_w is None:
        raise RuntimeError(
            f"Optimizer failed from every starting point: {message}. Usually "
            f"means the constraints are infeasible -- check max_weight."
        )

    w = np.clip(best_w, 0.0, max_weight)
    return w / w.sum()


def _starting_points(n_assets: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [np.full(n_assets, 1.0 / n_assets)]
    starts.extend(np.eye(n_assets))
    starts.extend(rng.dirichlet(np.ones(n_assets), size=8))
    return starts


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def optimize(
    panel: ReturnPanel,
    objective: Objective = Objective.MAX_SHARPE,
    max_weight: float = 1.0,
    risk_free: pd.Series | float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> OptimizationResult:
    """Find the best allocation over the panel's period for one objective.

    `tolerance` sets what counts as near-optimal for the reported range: 0.02
    means within 2% of the best value. `n_samples` is the search budget for the
    path-dependent objectives; it has no effect on the exact ones beyond
    supplying their near-optimal range.
    """
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must be between 0 and 1, got {tolerance}")

    assets = panel.assets
    n = len(assets)
    cov = panel.ann_cov().to_numpy()

    samples = sample_allocations(n, n_samples, max_weight, seed)
    measured = _measure_samples(panel, samples, risk_free)

    if objective in _EXACTLY_SOLVABLE:
        method = Method.EXACT
        if objective is Objective.MIN_VOLATILITY:
            weights_array = _solve_smooth(
                lambda w: w @ cov @ w, n, max_weight, _starting_points(n, seed)
            )
        else:
            # The numerator must be the same quantity `PortfolioStats.sharpe`
            # reports: the annualised *arithmetic* mean of excess returns. An
            # earlier version maximised a geometric-return numerator here, so
            # the solver optimised one thing and was scored on another -- which
            # let a constrained answer appear to beat an unconstrained one, an
            # impossibility that only showed up as a negative constraint cost.
            ppy = panel.periods_per_year
            if isinstance(risk_free, pd.Series):
                rf_periodic = risk_free.reindex(panel.returns.index).to_numpy()
            elif risk_free is None:
                rf_periodic = (
                    panel.returns["cash"].to_numpy() if "cash" in assets else 0.0
                )
            else:
                rf_periodic = (1.0 + float(risk_free)) ** (1.0 / ppy) - 1.0

            mean_returns = panel.returns.mean().to_numpy()
            mean_rf = float(np.mean(rf_periodic))

            def negative_sharpe(w: np.ndarray) -> float:
                vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
                if vol < MIN_MEANINGFUL_VOL:
                    return 0.0
                excess = (float(w @ mean_returns) - mean_rf) * ppy
                return -excess / vol

            weights_array = _solve_smooth(
                negative_sharpe, n, max_weight, _starting_points(n, seed)
            )
    else:
        method = Method.SAMPLED
        column = _COLUMN_FOR[objective]

        # An unbounded ratio is degenerate, not optimal. A portfolio sitting in
        # cash never falls below cash, so its downside deviation is zero and
        # its Sortino is infinite -- arithmetically true and financially
        # meaningless. Selecting it would hand the user a "best allocation"
        # that takes no risk, earns nothing, and scores infinity.
        finite = measured[np.isfinite(measured[column])]
        if finite.empty:
            raise RuntimeError(
                f"Every sampled allocation produced a degenerate {column}. "
                f"This usually means the panel has no risky asset."
            )

        best_row = (
            finite[column].idxmax()
            if (_MAXIMISE[objective] or objective is Objective.MIN_DRAWDOWN)
            else finite[column].idxmin()
        )
        weights_array = measured.loc[best_row, assets].to_numpy(dtype=float)
        weights_array = _refine(
            panel, weights_array, objective, max_weight, risk_free, seed
        )

    weights = pd.Series(weights_array, index=assets)
    stats = portfolio_stats(panel, weights, risk_free=risk_free)
    best_value = score(stats, objective)

    return OptimizationResult(
        weights=weights,
        stats=stats,
        objective=objective,
        method=method,
        near_optimal=_near_optimal(measured, objective, best_value, tolerance),
        tolerance=tolerance,
        n_samples=len(samples),
    )


def _refine(
    panel: ReturnPanel,
    start: np.ndarray,
    objective: Objective,
    max_weight: float,
    risk_free: pd.Series | float | None,
    seed: int,
    rounds: int = 4,
    per_round: int = 500,
) -> np.ndarray:
    """Local search around a sampled winner, with a shrinking step size.

    The sampled optimum is close but rarely exact, since random draws thin out
    quickly in higher dimensions. This walks the neighbourhood at decreasing
    scale, keeping any improvement.
    """
    rng = np.random.default_rng(seed + 1)
    n = len(start)

    best = start.copy()
    best_value = score(
        portfolio_stats(panel, pd.Series(best, index=panel.assets), risk_free=risk_free),
        objective,
    )

    step = 0.10
    for _ in range(rounds):
        perturbed = best + rng.normal(0.0, step, size=(per_round, n))
        perturbed = np.clip(perturbed, 0.0, None)
        totals = perturbed.sum(axis=1, keepdims=True)
        perturbed = np.divide(
            perturbed, totals, out=np.zeros_like(perturbed), where=totals > 0
        )
        perturbed = perturbed[perturbed.sum(axis=1) > 0.5]
        if max_weight < 1.0:
            perturbed = _apply_cap(perturbed, max_weight)

        measured = _measure_samples(panel, perturbed, risk_free)
        column = _COLUMN_FOR[objective]
        candidate_row = (
            measured[column].idxmax()
            if (_MAXIMISE[objective] or objective is Objective.MIN_DRAWDOWN)
            else measured[column].idxmin()
        )
        candidate_value = float(measured.loc[candidate_row, column])

        if is_better(candidate_value, best_value, objective):
            best = measured.loc[candidate_row, panel.assets].to_numpy(dtype=float)
            best_value = candidate_value

        step /= 3.0

    return best


def optimize_all(
    panel: ReturnPanel,
    max_weight: float = 1.0,
    risk_free: pd.Series | float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> dict[Objective, OptimizationResult]:
    """Solve every objective. Each answers a different question about risk."""
    return {
        objective: optimize(
            panel, objective, max_weight, risk_free, tolerance, n_samples, seed
        )
        for objective in Objective
    }


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------

def efficient_frontier(
    panel: ReturnPanel,
    n_points: int = 30,
    max_weight: float = 1.0,
    risk_free: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Lowest volatility achievable at each level of expected return.

    The curve rather than a point: the user chooses where on the trade-off to
    sit instead of being handed one answer. Points the solver cannot reach are
    dropped rather than reported with wrong numbers, so the result may be
    shorter than `n_points`.
    """
    assets = panel.assets
    n = len(assets)
    mu = panel.ann_return().to_numpy()
    cov = panel.ann_cov().to_numpy()

    floor_weights = _solve_smooth(
        lambda w: w @ cov @ w, n, max_weight, _starting_points(n)
    )
    floor = float(floor_weights @ mu)

    # Highest reachable return: fill the best assets in order, up to the cap.
    order = np.argsort(mu)[::-1]
    remaining, ceiling_weights = 1.0, np.zeros(n)
    for i in order:
        take = min(max_weight, remaining)
        ceiling_weights[i] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    ceiling = float(ceiling_weights @ mu)

    if ceiling <= floor + 1e-12:
        raise ValueError(
            "No return range to sweep: the highest reachable return equals the "
            "minimum-variance return. This happens when max_weight leaves only "
            "one feasible allocation."
        )

    rows = []
    for target in np.linspace(floor, ceiling, n_points):
        constraints = [
            _budget_constraint(),
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        result = minimize(
            lambda w: w @ cov @ w,
            np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n,
            constraints=constraints,
            options={"maxiter": _MAX_ITER, "ftol": _FTOL},
        )
        if not result.success:
            continue

        w = np.clip(result.x, 0.0, max_weight)
        w = w / w.sum()
        achieved = float(w @ mu)
        if abs(achieved - target) > 1e-5:
            continue

        stats = portfolio_stats(
            panel, pd.Series(w, index=assets), risk_free=risk_free
        )
        row = {
            "expected_return": achieved,
            "volatility": stats.volatility,
            "sharpe": stats.sharpe,
            "sortino": stats.sortino,
            "max_drawdown": stats.max_drawdown,
        }
        row.update(dict(zip(assets, w)))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("expected_return", ignore_index=True)
