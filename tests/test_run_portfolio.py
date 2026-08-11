"""Tests for the weight parsing in scripts/run_portfolio.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_portfolio import ORDER, SAMPLES, parse_weights, to_fractions  # noqa: E402


def test_every_sample_allocation_sums_to_one_hundred():
    """A sample that does not sum to 100 would be silently renormalised or
    rejected deep in the stack, long after the typo was made."""
    for sample in SAMPLES:
        assert sum(sample) == pytest.approx(100.0), f"{sample} sums to {sum(sample)}"


def test_every_sample_has_one_weight_per_asset():
    for sample in SAMPLES:
        assert len(sample) == len(ORDER)


def test_samples_are_structurally_different():
    """The point of the set is to exercise the maths across different shapes,
    not to compare near-identical portfolios."""
    equity_weights = {sample[0] for sample in SAMPLES}
    assert len(equity_weights) >= 3


def test_parse_accepts_slashes_and_commas():
    assert parse_weights("60/20/10/5/5") == (60.0, 20.0, 10.0, 5.0, 5.0)
    assert parse_weights("60,20,10,5,5") == (60.0, 20.0, 10.0, 5.0, 5.0)


def test_parse_accepts_decimals():
    assert parse_weights("55/12.5/20/7.5/5") == (55.0, 12.5, 20.0, 7.5, 5.0)


def test_parse_rejects_a_total_other_than_one_hundred():
    """The real failure this catches: 55/12.5/0.2/0.075/0.05 sums to 67.8,
    which is what a decimal formatting slip looks like."""
    with pytest.raises(ValueError, match="sum to"):
        parse_weights("55/12.5/0.2/0.075/0.05")


def test_parse_rejects_the_wrong_count():
    with pytest.raises(ValueError, match="Expected"):
        parse_weights("60/20/20")


def test_fractions_sum_to_one():
    weights = to_fractions((55.0, 12.5, 20.0, 7.5, 5.0))
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == set(ORDER)
