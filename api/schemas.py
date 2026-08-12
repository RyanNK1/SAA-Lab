"""Request and response shapes for the API.

Kept separate from the endpoints so the contract is readable in one place, and
so validation failures are caught at the boundary with a message naming the
field rather than surfacing as an obscure error from deep in the engine.

Percentages are fractions everywhere -- 0.06 for 6% -- matching the core
modules. The most likely caller mistake is sending 6, so the bounds reject it
rather than silently solving an impossible mandate.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from core.config import Objective, Rebalance

MAX_SAMPLES = 100_000
INTERACTIVE_SAMPLES = 4_000


class ConstraintSpec(BaseModel):
    """Policy limits, as a caller supplies them."""

    caps: dict[str, float] = Field(default_factory=dict)
    floors: dict[str, float] = Field(default_factory=dict)
    group_caps: dict[str, float] = Field(
        default_factory=dict,
        description="Named group -> maximum, e.g. {'growth': 0.6}",
    )
    group_floors: dict[str, float] = Field(default_factory=dict)

    @field_validator("caps", "floors", "group_caps", "group_floors")
    @classmethod
    def _fractions_only(cls, value: dict[str, float]) -> dict[str, float]:
        for name, limit in value.items():
            if not 0.0 <= limit <= 1.0:
                raise ValueError(
                    f"limit for {name!r} must be a fraction between 0 and 1 "
                    f"(0.2 for 20%), got {limit}"
                )
        return value


class BaseRequest(BaseModel):
    """Settings every request shares."""

    start: str | None = None
    end: str | None = None
    gold_weight: float = Field(
        0.5, ge=0.0, le=1.0, description="1.0 is all gold, 0.0 all ex-gold"
    )
    assets: list[str] | None = Field(
        None, description="Subset to allocate across; all five if omitted"
    )
    rebalance: Rebalance = Rebalance.MONTHLY
    cost_bps: float = Field(0.0, ge=0.0, le=500.0)
    samples: int = Field(INTERACTIVE_SAMPLES, ge=100, le=MAX_SAMPLES)


class MeasureRequest(BaseRequest):
    """Measure one allocation the caller supplies."""

    weights: dict[str, float]

    @field_validator("weights")
    @classmethod
    def _must_sum_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("weights cannot be empty")
        total = sum(value.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"weights must sum to 1.0 (they are fractions, not "
                f"percentages); got {total:.6f}"
            )
        return value


class OptimizeRequest(BaseRequest):
    """Find the best allocation on one measure."""

    objective: Objective = Objective.MAX_SHARPE
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    tolerance: float = Field(
        0.02,
        gt=0.0,
        lt=1.0,
        description="What counts as equivalent to the best, e.g. 0.02 for 2%",
    )


class MandateRequest(BaseRequest):
    """Solve a mandate: what must be achieved, within what limits."""

    target_return: float | None = Field(None, gt=-1.0, lt=5.0)
    max_volatility: float | None = Field(None, gt=0.0, lt=5.0)
    max_drawdown: float | None = Field(None, ge=-1.0, le=0.0)
    max_recovery_months: int | None = Field(None, ge=1)
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    rank_by: str = "max_drawdown"
    limit: int = Field(20, ge=1, le=500)

    @model_validator(mode="after")
    def _needs_a_requirement(self) -> MandateRequest:
        if all(
            value is None
            for value in (
                self.target_return,
                self.max_volatility,
                self.max_drawdown,
                self.max_recovery_months,
            )
        ):
            raise ValueError(
                "A mandate needs at least one requirement. With none, every "
                "allocation qualifies and there is nothing to solve."
            )
        return self


class SweepRequest(MandateRequest):
    """Find where a return target stops being reachable."""

    target_from: float = Field(0.02, gt=-1.0, lt=5.0)
    target_to: float = Field(0.12, gt=-1.0, lt=5.0)
    target_step: float = Field(0.01, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _range_is_sensible(self) -> SweepRequest:
        if self.target_to <= self.target_from:
            raise ValueError("target_to must be above target_from")
        span = self.target_to - self.target_from
        if span / self.target_step > 50:
            raise ValueError(
                f"That range and step would need "
                f"{int(span / self.target_step)} solves. Use a coarser step."
            )
        return self


class PeriodsRequest(BaseRequest):
    """Compare the same question across regimes."""

    objective: Objective = Objective.MAX_SHARPE
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    rolling_years: int | None = Field(
        None, ge=1, le=20, description="Rolling windows instead of named regimes"
    )


class RobustMandateRequest(MandateRequest):
    """Require a mandate to hold in every regime, not merely overall."""

    rolling_years: int | None = Field(None, ge=1, le=20)
