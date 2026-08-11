"""Tests for the review logic in scripts/build_dataset.py.

The script is not an importable package, so the scripts directory is added to
the path once here rather than inside each test. Kept in its own module
because it tests a script, not a core module.

The review function matters more than it looks: it is the difference between
a validation report a human has to read carefully and one that says plainly
when something is wrong. A splice error produces a single catastrophic month
in a table of otherwise normal numbers -- easy to miss by eye, trivial to
catch by rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_dataset import review  # noqa: E402


def _report(**overrides) -> pd.DataFrame:
    """A single-asset report that passes review, with fields overridable."""
    row = {
        "n_obs": 200,
        "zero_return_months": 0,
        "min_return": -0.15,
        "max_return": 0.12,
        "ann_return": 0.08,
        "ann_vol": 0.16,
    }
    row.update(overrides)
    return pd.DataFrame({"equity": row}).T


def test_plausible_report_raises_nothing():
    assert review(_report()) == []


def test_catastrophic_month_is_flagged_as_a_splice_error():
    """A month near -100% is the chain-link failure signature, and the warning
    should say so rather than just noting an odd number."""
    warnings = review(_report(min_return=-0.99))
    assert warnings
    assert any("splice" in w for w in warnings)


def test_implausible_gain_is_flagged():
    assert any("implausibly large" in w for w in review(_report(max_return=0.95)))


def test_zero_volatility_is_flagged():
    """A flat series means a construction error, not a calm asset."""
    assert any("volatility" in w for w in review(_report(ann_vol=0.0)))


def test_extreme_volatility_is_flagged():
    assert any("volatility" in w for w in review(_report(ann_vol=1.5)))


def test_many_stale_months_are_flagged():
    assert any("zero return" in w for w in review(_report(zero_return_months=40)))


def test_a_few_stale_months_are_tolerated():
    """Occasional repeated quotes are normal; warning on them would train the
    reader to ignore warnings."""
    assert review(_report(zero_return_months=2)) == []


def test_normal_market_extremes_are_tolerated():
    """October 2008 was about -20% for equities. The bounds must not fire on
    real crashes, only on construction errors."""
    assert review(_report(min_return=-0.22, max_return=0.15)) == []


def test_every_asset_is_reviewed_not_just_the_first():
    report = pd.concat([_report(), _report(min_return=-0.99).rename(index={"equity": "gold"})])
    warnings = review(report)
    assert any("gold" in w for w in warnings)


def test_multiple_problems_are_all_reported():
    warnings = review(_report(min_return=-0.99, ann_vol=0.0, zero_return_months=50))
    assert len(warnings) >= 3
