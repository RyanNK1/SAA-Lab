"""Tests for scripts/run_rebalance.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_rebalance import DEFAULT_WEIGHTS, ORDER, parse_weights  # noqa: E402


def test_default_allocation_sums_to_one_hundred():
    assert sum(DEFAULT_WEIGHTS) == pytest.approx(100.0)


def test_order_matches_the_allocatable_universe():
    """The script's asset order must track config, or a user typing weights
    positionally would silently allocate to the wrong buckets."""
    from core import config

    assert tuple(ORDER) == config.ALLOCATABLE_KEYS


def test_parse_rejects_a_total_other_than_one_hundred():
    with pytest.raises(ValueError, match="sum to"):
        parse_weights("55/12.5/0.2/0.075/0.05")


def test_parse_rejects_the_wrong_count():
    with pytest.raises(ValueError, match="Expected"):
        parse_weights("60/20/20")


def test_parse_accepts_slashes_and_commas():
    assert parse_weights("60/10/20/7.5/2.5") == (60.0, 10.0, 20.0, 7.5, 2.5)
    assert parse_weights("60,10,20,7.5,2.5") == (60.0, 10.0, 20.0, 7.5, 2.5)
