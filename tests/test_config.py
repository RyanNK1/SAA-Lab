"""Tests for the repo skeleton and configuration.

These look trivial, and they are. Their job is to fail loudly the moment the
project structure breaks -- a renamed asset key, a duplicate entry, an import
that stops resolving. Those failures are cheap to fix now and expensive to
diagnose once ten modules depend on them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core import config


def test_package_imports():
    """The package resolves and exposes what everything else will import."""
    assert config.MONTHS_PER_YEAR == 12
    assert config.ASSET_KEYS


def test_asset_keys_are_unique():
    assert len(config.ASSET_KEYS) == len(set(config.ASSET_KEYS))


def test_asset_keys_are_identifier_safe():
    """Keys become column names, JSON fields and URL parameters. Anything with
    a space, dash or capital will eventually break one of those."""
    for key in config.ASSET_KEYS:
        assert key.isidentifier(), f"{key!r} is not a safe identifier"
        assert key.islower(), f"{key!r} should be lowercase"


def test_every_asset_has_a_label_and_caveat():
    """No proxy is the thing it represents. The difference is documented at
    the point of definition, not somewhere else."""
    for item in config.UNIVERSE:
        assert item.label.strip(), f"{item.key} has no label"
        assert item.caveat.strip(), f"{item.key} has no caveat"


def test_asset_lookup_works_and_fails_clearly():
    assert config.asset("equity").label == "Public equity"
    with pytest.raises(KeyError, match="Unknown asset class"):
        config.asset("bitcoin")


def test_enums_cover_the_agreed_scope():
    """The design fixed these choices. If one disappears, a feature was
    silently dropped."""
    assert {c.value for c in config.Currency} >= {"USD", "GBP"}
    assert {r.value for r in config.Rebalance} >= {
        "never",
        "monthly",
        "quarterly",
        "annual",
        "threshold",
    }
    assert {o.value for o in config.Objective} >= {
        "max_sharpe",
        "max_sortino",
        "min_volatility",
        "min_drawdown",
    }


def test_defaults_are_members_of_their_enums():
    assert config.DEFAULT_CURRENCY in config.Currency
    assert config.DEFAULT_REBALANCE in config.Rebalance


def test_cost_and_threshold_are_sane():
    assert 0.0 <= config.DEFAULT_COST_BPS <= 100.0
    assert 0.0 < config.DEFAULT_THRESHOLD_BAND < 1.0


# ---------------------------------------------------------------------------
# Config must describe what the code actually builds
# ---------------------------------------------------------------------------

def test_config_covers_every_asset_the_data_layer_produces():
    """Config drifting from reality is the failure it exists to prevent. An
    asset the pipeline builds but config does not know about will be missing
    its label and caveat everywhere they are shown."""
    from core.market_data import DIRECT_TICKERS

    built = {"equity", "cash", *DIRECT_TICKERS}
    declared = set(config.ASSET_KEYS)

    assert built == declared, (
        f"built but undeclared: {sorted(built - declared)}; "
        f"declared but never built: {sorted(declared - built)}"
    )


def test_no_caveat_is_still_a_placeholder():
    """A placeholder caveat means a real proxy decision was made somewhere and
    never written down where it is shown to the user."""
    unfinished = [a.key for a in config.UNIVERSE if "placeholder" in a.caveat.lower()]
    assert not unfinished, f"still placeholders: {unfinished}"


def test_sleeve_components_are_real_assets():
    for key in config.SLEEVE_COMPONENTS:
        assert key in config.ASSET_KEYS


def test_allocatable_keys_match_the_sleeve_construction():
    """After the sleeve is built, the user allocates across these. The raw
    components must not appear -- they were replaced."""
    from core.sleeve import SLEEVE

    assert config.SLEEVE_KEY == SLEEVE
    for component in config.SLEEVE_COMPONENTS:
        assert component not in config.ALLOCATABLE_KEYS

    non_sleeve = set(config.ASSET_KEYS) - set(config.SLEEVE_COMPONENTS)
    assert non_sleeve | {config.SLEEVE_KEY} == set(config.ALLOCATABLE_KEYS)


def test_allocatable_order_matches_the_script():
    from run_portfolio import ORDER

    assert tuple(ORDER) == config.ALLOCATABLE_KEYS
