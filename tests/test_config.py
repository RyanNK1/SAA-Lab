"""Tests for the repo skeleton and configuration.

These look trivial, and they are. Their job is to fail loudly the moment the
project structure breaks -- a renamed asset key, a duplicate entry, an import
that stops resolving. Those failures are cheap to fix now and expensive to
diagnose once ten modules depend on them.
"""

from __future__ import annotations

import pytest

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
    assert config.asset("equity").label == "Global equities"
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
