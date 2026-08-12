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

    `caveat` and `proxy` are both required. Every proxy differs from the thing
    it represents, and a user reading "private equity" deserves to know they
    are looking at a small-cap index. Naming the instrument next to the
    definition means the interface can always show it, rather than the
    difference living in a document nobody opens.
    """

    key: str
    label: str
    proxy: str
    caveat: str


UNIVERSE: tuple[AssetClass, ...] = (
    AssetClass(
        key="equity",
        label="Public equity",
        proxy="MSCI ACWI (ACWI), spliced to a SPY/EFA/EEM blend before 2008",
        caveat=(
            "ACWI from 2008-03; before that a fixed 55/33/12 blend of SPY, "
            "EFA and EEM chain-linked behind it, because ACWI did not exist "
            "and the financial crisis would otherwise fall outside the data."
        ),
    ),
    AssetClass(
        key="fixed_income",
        label="Fixed income",
        proxy="Bloomberg US Aggregate Bond Index (AGG)",
        caveat=(
            "AGG: the US aggregate, not global. No non-USD exposure, so this "
            "understates the volatility a global bond sleeve would carry."
        ),
    ),
    AssetClass(
        key="private_equity",
        label="Private equity",
        proxy="Russell 2000 small cap (IWM), as a replication proxy",
        caveat=(
            "IWM (Russell 2000) as a replication proxy -- PE returns are "
            "largely reproducible with small-cap value plus leverage. "
            "Correlates around 0.9 with public equity, so this sleeve is not "
            "the diversifier its label implies; the risk contribution table "
            "exists to make that visible. Listed-PE vehicles were rejected: "
            "they hold leveraged PE managers and returned about 2.6% a year "
            "since 2006, nothing like institutional PE."
        ),
    ),
    AssetClass(
        key="gold",
        label="Gold",
        proxy="LBMA gold via SPDR Gold Shares (GLD)",
        caveat=(
            "GLD, a physical gold trust, net of a 0.40% fee. A component of "
            "the commodities sleeve rather than an allocatable bucket on its "
            "own -- the slider sets its share."
        ),
    ),
    AssetClass(
        key="commodities_ex_gold",
        label="Commodities ex-gold",
        proxy="DBIQ Optimum Yield Diversified Commodity Index (DBC)",
        caveat=(
            "DBC, a broad futures basket carrying roll-yield effects that "
            "spot commodity prices do not. The other component of the "
            "commodities sleeve. Behaves very differently from gold: it fell "
            "with equities in 2008 rather than rising."
        ),
    ),
    AssetClass(
        key="cash",
        label="Cash",
        proxy="13-week US Treasury bill rate (^IRX), compounded",
        caveat=(
            "The 13-week T-bill rate compounded into an index. Quoted as a "
            "discount rate, which slightly understates a true bond-equivalent "
            "yield -- immaterial for a small sleeve. Went briefly negative in "
            "late 2015 and March 2020, which is real history, not an error."
        ),
    ),
)

# The sleeve the user actually allocates to. Gold and commodities ex-gold are
# combined into this before any portfolio is measured, so the optimizer never
# sees them separately.
SLEEVE_KEY = "commodities"
SLEEVE_COMPONENTS = ("gold", "commodities_ex_gold")

# What a user can put a weight on, after the sleeve is built.
ALLOCATABLE_KEYS: tuple[str, ...] = (
    "equity",
    "fixed_income",
    "private_equity",
    SLEEVE_KEY,
    "cash",
)

ASSET_KEYS: tuple[str, ...] = tuple(a.key for a in UNIVERSE)

# The commodities sleeve is what a user actually allocates to. It does not
# exist in the raw data -- it is built from its two components at request time,
# according to the slider -- so it is described here rather than in UNIVERSE,
# which lists the series the data layer produces.
SLEEVE_ASSET = AssetClass(
    key=SLEEVE_KEY,
    label="Commodities",
    proxy="Gold (GLD) and commodities ex-gold (DBC), blended by the slider",
    caveat=(
        "One bucket holding two things that behave very differently: gold rose "
        "through the 2008 crisis while broad commodities fell with equities. "
        "The slider decides which of those the sleeve mostly is."
    ),
)


def allocatable_universe() -> tuple[AssetClass, ...]:
    """The buckets a weight can be put on, after the sleeve is built.

    Gold and commodities ex-gold are replaced by the single sleeve, because by
    the time an allocation is made they no longer exist separately.
    """
    return tuple(
        SLEEVE_ASSET if item.key == SLEEVE_COMPONENTS[0] else item
        for item in UNIVERSE
        if item.key != SLEEVE_COMPONENTS[1]
    )


def asset(key: str) -> AssetClass:
    """Look up one asset class by key, with a useful error if it is missing."""
    for item in UNIVERSE:
        if item.key == key:
            return item
    raise KeyError(f"Unknown asset class {key!r}. Known: {list(ASSET_KEYS)}")
