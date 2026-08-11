"""Central configuration for the whole project.

Every constant that more than one module needs lives here. The alternative --
each module carrying its own copy of "12 months in a year" or its own list of
asset classes -- is how a rename in one place silently breaks another.

The universe below is a starting point and is expected to change. Nothing in
the codebase should hardcode an asset name; read from here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

MONTHS_PER_YEAR = 12
TRADING_DAYS_PER_YEAR = 252

# Minimum observations before a period's statistics are reported without a
# caveat. Not a hard block -- the user may legitimately ask about a short
# window -- but the interface should say when it is below this.
MIN_OBSERVATIONS = 24


class Currency(str, Enum):
    """Reporting currency. An asset's return depends on who is holding it."""

    USD = "USD"
    GBP = "GBP"


DEFAULT_CURRENCY = Currency.USD


class Rebalance(str, Enum):
    """How often an allocation is corrected back to its target weights."""

    NEVER = "never"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    THRESHOLD = "threshold"


DEFAULT_REBALANCE = Rebalance.ANNUAL

# Drift, in percentage points, that triggers a trade under THRESHOLD.
DEFAULT_THRESHOLD_BAND = 0.05

# One-way transaction cost charged on turnover, in basis points.
DEFAULT_COST_BPS = 10.0


class Objective(str, Enum):
    """What 'best allocation' means for a given request."""

    MAX_SHARPE = "max_sharpe"
    MAX_SORTINO = "max_sortino"
    MIN_VOLATILITY = "min_volatility"
    MIN_DRAWDOWN = "min_drawdown"


# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssetClass:
    """One bucket a user can allocate to.

    `caveat` is deliberately a required field. Every proxy differs from the
    thing it represents, and the difference belongs next to the definition
    rather than in a document nobody opens.
    """

    key: str
    label: str
    caveat: str


UNIVERSE: tuple[AssetClass, ...] = (
    AssetClass(
        key="equity",
        label="Global equities",
        caveat="Placeholder. Proxy and history window not yet chosen.",
    ),
    AssetClass(
        key="fixed_income",
        label="Fixed income",
        caveat="Placeholder. Global vs US aggregate not yet decided.",
    ),
    AssetClass(
        key="private_equity",
        label="Private equity",
        caveat=(
            "US small cap (Russell 2000), used as a replication proxy: PE "
            "returns are largely reproducible with small-cap value plus "
            "leverage. Correlates ~0.9 with public equity, so this sleeve is "
            "not the diversifier its label implies. Listed-PE vehicles were "
            "rejected as a proxy -- they hold leveraged PE *managers* and "
            "returned ~2.6% a year since 2006, nothing like institutional PE."
        ),
    ),
    AssetClass(
        key="gold",
        label="Gold",
        caveat="Placeholder. Spot vs futures not yet decided.",
    ),
    AssetClass(
        key="cash",
        label="Cash",
        caveat="Placeholder. Rate series and currency dependence not yet set.",
    ),
)

ASSET_KEYS: tuple[str, ...] = tuple(a.key for a in UNIVERSE)


def asset(key: str) -> AssetClass:
    """Look up one asset class by key, with a useful error if it is missing."""
    for item in UNIVERSE:
        if item.key == key:
            return item
    raise KeyError(f"Unknown asset class {key!r}. Known: {list(ASSET_KEYS)}")
