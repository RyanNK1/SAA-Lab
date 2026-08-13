"""HTTP interface to the allocation engine.

A thin layer. Every endpoint translates a request into a call on `core`,
converts the result to JSON, and does nothing else -- no calculation lives
here. That separation is deliberate: the engine is covered by hundreds of
tests that never touch HTTP, and it stays that way.

Two operational decisions worth knowing:

**The dataset is loaded once at startup.** It is small and never changes
during a run, so re-reading it per request would be waste. A data refresh
means a restart, which is acceptable for how this is deployed.

**Sample budgets default low.** A 20,000-sample solve takes seconds, which is
fine for a considered run and far too slow behind a slider. Interactive
requests default to 4,000 and the caller can raise it. The cross-period
endpoints are slow whatever the budget, and are separated so a frontend can
treat them differently.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ConstraintSpec,
    TrackRequest,
    MandateRequest,
    MeasureRequest,
    OptimizeRequest,
    PeriodsRequest,
    RobustMandateRequest,
    SweepRequest,
)
from core.config import (
    SLEEVE_COMPONENTS,
    UNIVERSE,
    Objective,
    Rebalance,
    allocatable_universe,
    asset,
)
from core.constraints import Constraints, GroupLimit, optimize_constrained
from core.mandate import (
    RANKABLE,
    Mandate,
    frontier_of_mandates,
    solve_mandate,
    solve_mandate_across_periods,
)
from core.optimize import efficient_frontier, optimize
from core.panels import ReturnPanel
from core.periods import (
    compare_periods,
    consensus_allocation,
    cross_period_performance,
    hindsight_premium,
    resolve_periods,
    rolling_periods,
    weight_stability,
)
from core.portfolio import portfolio_stats, risk_contributions
from core.rebalance import RebalanceSpec, measure_path, simulate
from core.sleeve import SLEEVE, build_sleeve, sleeve_sensitivity

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "monthly_returns.csv"

# Groups a caller can name in a constraint. Defined here rather than accepted
# as arbitrary asset lists, so a request cannot invent a grouping the rest of
# the application does not recognise.
GROUPS: dict[str, tuple[str, ...]] = {
    "growth": ("equity", "private_equity"),
    "risky": ("equity", "fixed_income", "private_equity", SLEEVE),
    "real_assets": (SLEEVE,),
}

app = FastAPI(
    title="SAA Lab",
    description=(
        "Strategic asset allocation over historical windows. Every answer is "
        "hindsight -- what would have been best over the period chosen. That "
        "is exactly answerable and is not a forecast."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_PANEL: ReturnPanel | None = None


def load_panel() -> ReturnPanel:
    """The full dataset, loaded once."""
    global _PANEL
    if _PANEL is None:
        if not DATA.exists():
            raise RuntimeError(
                f"Dataset missing at {DATA}. Run scripts/build_dataset.py and "
                f"commit the result before starting the server."
            )
        frame = pd.read_csv(DATA, index_col=0, parse_dates=True).drop(
            columns=["currency"], errors="ignore"
        )
        _PANEL = ReturnPanel(frame)
    return _PANEL


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

def _clean(value: Any) -> Any:
    """Make a value JSON-safe.

    Infinity is meaningful here -- it is how "never recovered from the worst
    drawdown" is encoded -- but it is not valid JSON. It becomes null, with the
    field name making the meaning clear to a caller.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _clean(value) for key, value in row.items()}
        for _, row in frame.iterrows()
    ]


def _series_to_dict(series: pd.Series) -> dict[str, Any]:
    return {str(key): _clean(value) for key, value in series.items()}


def prepare(request: Any) -> tuple[ReturnPanel, pd.Series | None]:
    """Apply the period, asset selection and sleeve settings from a request.

    Returns the panel the engine should work on and its cash series, if any.
    Every endpoint starts here, so the meaning of a period or an asset subset
    cannot drift between them.
    """
    try:
        panel = load_panel()
    except RuntimeError as error:
        # A missing dataset is an operational fault, not a bad request. Left
        # unhandled it surfaces as a 500 with a stack trace, which tells a
        # caller nothing about what to do.
        raise HTTPException(status_code=503, detail=str(error)) from error

    if request.start or request.end:
        try:
            panel = panel.between(
                request.start or panel.start, request.end or panel.end
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    panel = build_sleeve(panel, request.gold_weight)

    if request.assets:
        unknown = [a for a in request.assets if a not in panel.assets]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown assets {unknown}. Available: {panel.assets}",
            )
        try:
            panel = panel.select(request.assets)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    cash = panel.returns["cash"] if "cash" in panel.assets else None
    return panel, cash


def build_constraints(spec: ConstraintSpec, assets: list[str]) -> Constraints:
    """Turn a request's limits into the engine's constraint object."""
    groups = []
    for name in set(spec.group_caps) | set(spec.group_floors):
        if name not in GROUPS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown group {name!r}. Available: {sorted(GROUPS)}",
            )
        members = tuple(a for a in GROUPS[name] if a in assets)
        if not members:
            continue
        groups.append(
            GroupLimit(
                name=name,
                assets=members,
                maximum=spec.group_caps.get(name),
                minimum=spec.group_floors.get(name),
            )
        )

    try:
        constraints = Constraints(
            caps={k: v for k, v in spec.caps.items() if k in assets},
            floors={k: v for k, v in spec.floors.items() if k in assets},
            groups=tuple(groups),
        )
        constraints.validate(assets)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return constraints


def _rebalance_spec(request: Any) -> RebalanceSpec:
    return RebalanceSpec(schedule=request.rebalance, cost_bps=request.cost_bps)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    try:
        panel = load_panel()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok", "months": len(panel)}


@app.get("/meta")
def meta() -> dict[str, Any]:
    """Everything a frontend needs to build its controls."""
    panel = load_panel()

    return {
        "coverage": {
            "start": panel.start.strftime("%Y-%m-%d"),
            "end": panel.end.strftime("%Y-%m-%d"),
            "months": len(panel),
        },
        # The buckets a weight can actually be placed on. Gold and commodities
        # ex-gold do not appear: by the time an allocation is made they have
        # been combined into the sleeve, so offering them as separate choices
        # would present controls the engine rejects.
        "assets": [
            {
                "key": item.key,
                "label": item.label,
                "proxy": item.proxy,
                "caveat": item.caveat,
                "allocatable": True,
            }
            for item in allocatable_universe()
        ],
        "sleeve": {
            "key": SLEEVE,
            "components": [
                {
                    "key": component,
                    "label": asset(component).label,
                    "proxy": asset(component).proxy,
                    "caveat": asset(component).caveat,
                }
                for component in SLEEVE_COMPONENTS
            ],
            "note": (
                "One bucket. The slider sets its internal split, and the "
                "sleeve's return, risk and correlations all change with it."
            ),
        },
        "objectives": [o.value for o in Objective],
        "rebalance_schedules": [r.value for r in Rebalance],
        "rankable": sorted(RANKABLE),
        "groups": {name: list(members) for name, members in GROUPS.items()},
        "regimes": [
            {"label": p.label, "start": p.start.strftime("%Y-%m-%d"),
             "end": p.end.strftime("%Y-%m-%d")}
            for p in resolve_periods(build_sleeve(panel, 0.5))
        ],
        "disclaimer": (
            "Every result is hindsight: what would have been best over the "
            "period chosen, knowing what happened in it. Not a forecast."
        ),
    }


@app.get("/assets/stats")
def asset_stats(start: str | None = None, end: str | None = None,
                gold_weight: float = 0.5) -> dict[str, Any]:
    """Per-asset return, risk and correlations for a period."""
    panel = load_panel()
    if start or end:
        try:
            panel = panel.between(start or panel.start, end or panel.end)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    sleeved = build_sleeve(panel, gold_weight)
    return {
        "months": len(sleeved),
        "start": sleeved.start.strftime("%Y-%m-%d"),
        "end": sleeved.end.strftime("%Y-%m-%d"),
        "assets": {
            asset: {
                "ann_return": _clean(sleeved.ann_return()[asset]),
                "ann_vol": _clean(sleeved.ann_vol()[asset]),
            }
            for asset in sleeved.assets
        },
        "correlations": {
            row: _series_to_dict(sleeved.corr().loc[row]) for row in sleeved.assets
        },
    }


@app.get("/sleeve/sensitivity")
def sleeve_slider(start: str | None = None, end: str | None = None,
                  steps: int = 11) -> dict[str, Any]:
    """How the commodities sleeve changes character across the slider."""
    panel = load_panel()
    if start or end:
        panel = panel.between(start or panel.start, end or panel.end)

    try:
        table = sleeve_sensitivity(panel, steps=max(2, min(steps, 51)))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {"steps": _frame_to_records(table)}


# ---------------------------------------------------------------------------
# Measuring and optimising
# ---------------------------------------------------------------------------

@app.post("/measure")
def measure(request: MeasureRequest) -> dict[str, Any]:
    """Statistics for one allocation the caller supplies."""
    panel, cash = prepare(request)
    spec = _rebalance_spec(request)

    try:
        if spec.schedule is Rebalance.MONTHLY and spec.cost_bps == 0.0:
            stats = portfolio_stats(panel, request.weights, risk_free=cash)
            turnover = cost = 0.0
        else:
            path = simulate(panel, request.weights, spec)
            stats = measure_path(panel, path, risk_free=cash)
            turnover, cost = path.total_turnover, path.total_cost

        contributions = risk_contributions(panel, request.weights)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "months": len(panel),
        "weights": {k: _clean(v) for k, v in request.weights.items()},
        "stats": {k: _clean(v) for k, v in stats.to_dict().items()},
        "drawdown": {
            "max_drawdown": _clean(stats.drawdown.max_drawdown),
            "peak": stats.drawdown.peak_date.strftime("%Y-%m-%d"),
            "trough": stats.drawdown.trough_date.strftime("%Y-%m-%d"),
            "recovered": stats.drawdown.recovered,
            "months_to_recover": _clean(stats.drawdown.months_to_recover),
            "months_underwater": _clean(stats.drawdown.months_underwater),
            "description": stats.drawdown.describe(),
        },
        "risk_contributions": _frame_to_records(contributions.reset_index(names="asset")),
        "turnover": _clean(turnover),
        "trading_cost": _clean(cost),
    }


@app.post("/optimize")
def optimize_endpoint(request: OptimizeRequest) -> dict[str, Any]:
    """Best allocation on one measure, with the near-optimal range."""
    panel, cash = prepare(request)
    constraints = build_constraints(request.constraints, panel.assets)

    try:
        if constraints.is_empty:
            result = optimize(
                panel,
                request.objective,
                risk_free=cash,
                tolerance=request.tolerance,
                n_samples=request.samples,
            )
        else:
            result = optimize_constrained(
                panel,
                request.objective,
                constraints,
                risk_free=cash,
                tolerance=request.tolerance,
                n_samples=request.samples,
            )
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "objective": request.objective.value,
        "method": result.method.value,
        "method_note": (
            "solved algebraically"
            if result.method.value == "exact"
            else (
                f"best of {result.n_samples:,} sampled allocations. Drawdown "
                f"and Sortino have no usable gradient, so this is a strong "
                f"candidate rather than a proven optimum."
            )
        ),
        "weights": _series_to_dict(result.weights),
        "stats": {k: _clean(v) for k, v in result.stats.to_dict().items()},
        "near_optimal_count": len(result.near_optimal),
        "ranges": _frame_to_records(result.ranges().reset_index(names="asset")),
        "tolerance": request.tolerance,
    }


@app.post("/frontier")
def frontier_endpoint(request: OptimizeRequest) -> dict[str, Any]:
    """The curve: lowest volatility reachable at each level of return."""
    panel, cash = prepare(request)

    try:
        curve = efficient_frontier(panel, n_points=30, risk_free=cash)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {"points": _frame_to_records(curve)}


# ---------------------------------------------------------------------------
# Mandates
# ---------------------------------------------------------------------------

def _mandate_from(request: MandateRequest, assets: list[str]) -> Mandate:
    constraints = build_constraints(request.constraints, assets)
    try:
        return Mandate(
            target_return=request.target_return,
            max_volatility=request.max_volatility,
            max_drawdown=request.max_drawdown,
            max_recovery_months=request.max_recovery_months,
            constraints=constraints,
            rebalance=_rebalance_spec(request),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/mandate")
def mandate_endpoint(request: MandateRequest) -> dict[str, Any]:
    """Which allocations met the mandate, or what would have to change."""
    panel, cash = prepare(request)
    mandate = _mandate_from(request, panel.assets)

    if request.rank_by not in RANKABLE:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot rank by {request.rank_by!r}. Available: {sorted(RANKABLE)}",
        )

    try:
        result = solve_mandate(
            panel, mandate, risk_free=cash, n_samples=request.samples
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    payload: dict[str, Any] = {
        "mandate": mandate.describe(),
        "feasible": result.feasible,
        "n_sampled": result.n_sampled,
        "n_qualifying": result.n_qualifying,
        "explanation": result.explain(),
    }

    if not result.feasible:
        payload["relaxations"] = [
            {
                "what": relaxation.what,
                "current": _clean(relaxation.current),
                "required": _clean(relaxation.required),
                "note": relaxation.note,
                "description": relaxation.describe(),
            }
            for relaxation in result.relaxations
        ]
        return payload

    ranked = result.ranked(request.rank_by, limit=request.limit)
    payload["ranked_by"] = request.rank_by

    # What the commodities weight means underneath. A user allocating 20% to
    # commodities at a 60/40 slider is holding 12% gold and 8% everything else,
    # and those two behave nothing alike -- so reporting only the sleeve total
    # hides the decision the slider actually made.
    if SLEEVE in panel.assets and not ranked.empty:
        sleeve_weight = float(ranked.iloc[0][SLEEVE])
        payload["sleeve_split"] = {
            "sleeve_weight": _clean(sleeve_weight),
            "gold_weight": _clean(request.gold_weight),
            "gold": _clean(sleeve_weight * request.gold_weight),
            "commodities_ex_gold": _clean(sleeve_weight * (1.0 - request.gold_weight)),
        }
    payload["allocations"] = _frame_to_records(ranked)
    payload["envelope"] = _frame_to_records(
        result.envelope().reset_index(names="asset")
    )
    payload["headroom"] = _frame_to_records(
        result.headroom().loc[ranked.index].reset_index(drop=True)
    )
    return payload


@app.post("/mandate/sweep")
def sweep_endpoint(request: SweepRequest) -> dict[str, Any]:
    """Where a return target stops being reachable within the budget."""
    panel, cash = prepare(request)
    constraints = build_constraints(request.constraints, panel.assets)

    if request.max_volatility is None:
        raise HTTPException(
            status_code=422,
            detail="A sweep needs max_volatility: it asks how much return that "
                   "budget could have supported.",
        )

    targets = [
        round(float(t), 6)
        for t in np.arange(
            request.target_from, request.target_to + 1e-9, request.target_step
        )
    ]

    try:
        frontier = frontier_of_mandates(
            panel,
            targets,
            request.max_volatility,
            constraints,
            risk_free=cash,
            n_samples=request.samples,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    reachable = frontier[frontier["feasible"]]
    return {
        "max_volatility": request.max_volatility,
        "points": _frame_to_records(frontier),
        "highest_reachable": (
            _clean(reachable["target"].max()) if len(reachable) else None
        ),
    }


@app.post("/mandate/across-periods")
def robust_mandate_endpoint(request: RobustMandateRequest) -> dict[str, Any]:
    """Allocations meeting the mandate in every regime, not merely overall.

    Slow. Meeting a mandate once is a hindsight result; meeting it through
    every regime is the harder claim, and it costs a full solve per period.
    """
    panel, _ = prepare(request)
    mandate = _mandate_from(request, panel.assets)

    periods = (
        rolling_periods(panel, years=request.rolling_years)
        if request.rolling_years
        else resolve_periods(panel)
    )
    if not periods:
        raise HTTPException(
            status_code=422, detail="No periods overlap the selected window"
        )

    try:
        result = solve_mandate_across_periods(
            panel, mandate, periods, n_samples=request.samples
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    payload: dict[str, Any] = {
        "mandate": mandate.describe(),
        "periods": result.period_labels,
        "qualified_per_period": {
            label: int(count) for label, count in result.survival_counts().items()
        },
        "n_survivors": len(result.survivors),
        "any_survivors": result.any_survivors,
        "explanation": result.explain(),
    }

    if result.any_survivors:
        survivors = result.survivors
        payload["envelope"] = _frame_to_records(
            pd.DataFrame(
                {
                    "min": survivors.min(),
                    "median": survivors.median(),
                    "max": survivors.max(),
                }
            ).reset_index(names="asset")
        )
        payload["examples"] = _frame_to_records(survivors.head(request.limit))

    return payload


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

@app.post("/periods/track")
def track_endpoint(request: TrackRequest) -> dict[str, Any]:
    """How specific allocations fared in each regime.

    Every allocation is measured in every period, which is a different question
    from optimising per period. A per-period optimum is a corner solution --
    everything into whatever happened to work -- and no committee would hold
    it, so what it endured is not informative. A candidate that came out of a
    mandate is something a person might actually own, and how it behaved
    through a crisis is worth knowing.
    """
    panel, _ = prepare(request)

    periods = (
        rolling_periods(panel, years=request.rolling_years)
        if request.rolling_years
        else resolve_periods(panel)
    )
    if request.periods:
        wanted = set(request.periods)
        periods = [p for p in periods if p.label in wanted]
    if not periods:
        raise HTTPException(
            status_code=422, detail="No periods match the selection"
        )

    spec = _rebalance_spec(request)
    rows = []

    for candidate in request.allocations:
        missing = set(panel.assets) - set(candidate.weights)
        extra = set(candidate.weights) - set(panel.assets)
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{candidate.label}: weights do not match the selected "
                    f"assets. Missing {sorted(missing)}, unexpected "
                    f"{sorted(extra)}."
                ),
            )

        measured = []
        for period in periods:
            window = panel.between(period.start, period.end)
            cash = window.returns["cash"] if "cash" in window.assets else None

            try:
                if spec.schedule is Rebalance.MONTHLY and spec.cost_bps == 0.0:
                    stats = portfolio_stats(window, candidate.weights, risk_free=cash)
                else:
                    path = simulate(window, candidate.weights, spec)
                    stats = measure_path(window, path, risk_free=cash)
            except (ValueError, KeyError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

            measured.append(
                {
                    "period": period.label,
                    "months": len(window),
                    "realised_return": _clean(stats.realised_return),
                    "volatility": _clean(stats.volatility),
                    "sharpe": _clean(stats.sharpe),
                    "sortino": _clean(stats.sortino),
                    "max_drawdown": _clean(stats.max_drawdown),
                    "months_underwater": _clean(stats.drawdown.months_underwater),
                }
            )

        returns = [m["realised_return"] for m in measured if m["realised_return"] is not None]
        rows.append(
            {
                "label": candidate.label,
                "weights": {k: _clean(v) for k, v in candidate.weights.items()},
                "by_period": measured,
                # The summary a person actually wants: where it did well, where
                # it hurt, and how often it lost money at all.
                "best_period": max(measured, key=lambda m: m["realised_return"])["period"]
                if returns
                else None,
                "worst_period": min(measured, key=lambda m: m["realised_return"])["period"]
                if returns
                else None,
                "negative_periods": sum(1 for r in returns if r < 0),
                "worst_drawdown": _clean(min(m["max_drawdown"] for m in measured)),
                "return_spread": _clean(max(returns) - min(returns)) if returns else None,
            }
        )

    return {
        "periods": [
            {
                "label": p.label,
                "start": p.start.strftime("%Y-%m-%d"),
                "end": p.end.strftime("%Y-%m-%d"),
                "months": len(panel.between(p.start, p.end)),
            }
            for p in periods
        ],
        "allocations": rows,
    }


@app.post("/periods/compare")
def compare_periods_endpoint(request: PeriodsRequest) -> dict[str, Any]:
    """The same question answered in each regime, and how much the answer moves."""
    panel, _ = prepare(request)
    constraints = build_constraints(request.constraints, panel.assets)

    periods = (
        rolling_periods(panel, years=request.rolling_years)
        if request.rolling_years
        else resolve_periods(panel)
    )
    if not periods:
        raise HTTPException(
            status_code=422, detail="No periods overlap the selected window"
        )

    try:
        table, results = compare_periods(
            panel,
            periods,
            objective=request.objective,
            constraints=constraints,
            n_samples=request.samples,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    matrix = cross_period_performance(panel, results, periods)
    premium = hindsight_premium(matrix)

    return {
        "objective": request.objective.value,
        "by_period": _frame_to_records(table.reset_index()),
        "stability": _frame_to_records(
            weight_stability(table, panel.assets).reset_index(names="asset")
        ),
        "cross_period": {
            "chosen_for": list(matrix.index),
            "measured_in": list(matrix.columns),
            "sharpe": [[_clean(v) for v in row] for row in matrix.to_numpy()],
            "note": (
                "The diagonal is in-sample and wins by construction: it was "
                "chosen knowing what happened. Read across a row to see "
                "whether it survived elsewhere."
            ),
        },
        "hindsight_premium": _frame_to_records(premium.reset_index()),
        "average_premium": _clean(premium["premium"].mean()),
        "consensus": _series_to_dict(consensus_allocation(table, panel.assets)),
    }
