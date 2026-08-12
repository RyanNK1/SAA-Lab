"""Tests for the HTTP layer.

These check translation, not calculation. The engine is covered by hundreds of
tests that never touch HTTP, and duplicating them here would mean two places
to update when the maths changes.

What matters at this boundary is: bad input is rejected with a message naming
the problem rather than reaching the engine and failing obscurely; results
survive JSON encoding without losing meaning; and the settings a caller sends
actually take effect.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app, load_panel  # noqa: E402

FAST = {"samples": 800}


@pytest.fixture(scope="module")
def client() -> TestClient:
    load_panel()  # fail here rather than inside a test if the dataset is missing
    return TestClient(app)


def _weights() -> dict[str, float]:
    return {
        "equity": 0.50,
        "fixed_income": 0.20,
        "private_equity": 0.15,
        "commodities": 0.10,
        "cash": 0.05,
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_health_reports_the_dataset_size(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["months"] > 100


def test_meta_gives_a_frontend_everything_it_needs(client):
    body = client.get("/meta").json()

    assert body["coverage"]["months"] > 100
    assert body["objectives"]
    assert body["rebalance_schedules"]
    assert body["rankable"]
    assert body["regimes"]
    assert "hindsight" in body["disclaimer"].lower()


def test_meta_marks_which_assets_can_be_allocated_to(client):
    """Gold and commodities ex-gold are sleeve components, not buckets a user
    puts a weight on. A frontend that showed them as allocatable would offer
    controls the engine rejects."""
    body = client.get("/meta").json()
    allocatable = {a["key"] for a in body["assets"] if a["allocatable"]}

    assert "gold" not in allocatable
    assert "commodities_ex_gold" not in allocatable
    assert "equity" in allocatable


def test_every_asset_carries_its_caveat(client):
    body = client.get("/meta").json()
    for asset in body["assets"]:
        assert asset["caveat"].strip()
        assert "placeholder" not in asset["caveat"].lower()


def test_asset_stats_returns_a_full_correlation_matrix(client):
    body = client.get("/assets/stats").json()
    assets = list(body["assets"])
    for row in assets:
        assert set(body["correlations"][row]) == set(assets)
        assert body["correlations"][row][row] == pytest.approx(1.0)


def test_sleeve_sensitivity_spans_the_slider(client):
    body = client.get("/sleeve/sensitivity?steps=6").json()
    weights = [step["gold_weight"] for step in body["steps"]]
    assert weights[0] == pytest.approx(0.0)
    assert weights[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def test_measure_returns_stats_drawdown_and_risk(client):
    body = client.post("/measure", json={"weights": _weights()}).json()

    assert body["stats"]["volatility"] > 0
    assert body["drawdown"]["max_drawdown"] < 0
    assert len(body["risk_contributions"]) == len(_weights())


def test_risk_contributions_sum_to_one(client):
    body = client.post("/measure", json={"weights": _weights()}).json()
    total = sum(row["pct_of_risk"] for row in body["risk_contributions"])
    assert total == pytest.approx(1.0, abs=1e-6)


def test_weights_not_summing_to_one_are_rejected(client):
    """The likeliest caller mistake, and it must not reach the engine."""
    response = client.post("/measure", json={"weights": {"equity": 0.5, "cash": 0.2}})
    assert response.status_code == 422
    assert "sum to 1.0" in str(response.json())


def test_a_rebalancing_schedule_changes_the_result(client):
    """If it did not, the setting would be decorative."""
    monthly = client.post(
        "/measure", json={"weights": _weights(), "rebalance": "monthly"}
    ).json()
    never = client.post(
        "/measure", json={"weights": _weights(), "rebalance": "never"}
    ).json()

    assert monthly["stats"]["realised_return"] != never["stats"]["realised_return"]


def test_trading_costs_reduce_the_return(client):
    free = client.post(
        "/measure",
        json={"weights": _weights(), "rebalance": "annual", "cost_bps": 0.0},
    ).json()
    costly = client.post(
        "/measure",
        json={"weights": _weights(), "rebalance": "annual", "cost_bps": 100.0},
    ).json()

    assert costly["stats"]["realised_return"] < free["stats"]["realised_return"]
    assert costly["trading_cost"] > 0


def test_a_period_selection_changes_the_answer(client):
    whole = client.post("/measure", json={"weights": _weights()}).json()
    crisis = client.post(
        "/measure",
        json={"weights": _weights(), "start": "2007-10-01", "end": "2009-02-28"},
    ).json()

    assert crisis["months"] < whole["months"]
    assert crisis["stats"]["max_drawdown"] != whole["stats"]["max_drawdown"]


def test_a_window_with_too_little_data_is_rejected(client):
    response = client.post(
        "/measure",
        json={"weights": _weights(), "start": "2010-01-01", "end": "2010-01-31"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Optimising
# ---------------------------------------------------------------------------

def test_optimize_reports_which_method_solved_it(client):
    """Exact and sampled answers are different kinds of claim, and a caller
    should be able to tell them apart."""
    sharpe = client.post(
        "/optimize", json={"objective": "max_sharpe", **FAST}
    ).json()
    drawdown = client.post(
        "/optimize", json={"objective": "min_drawdown", **FAST}
    ).json()

    assert sharpe["method"] == "exact"
    assert drawdown["method"] == "sampled"
    assert "candidate" in drawdown["method_note"]


def test_optimized_weights_are_valid(client):
    for objective in ("max_sharpe", "max_sortino", "min_volatility", "min_drawdown"):
        body = client.post("/optimize", json={"objective": objective, **FAST}).json()
        total = sum(body["weights"].values())
        assert total == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-9 for w in body["weights"].values())


def test_constraints_are_applied(client):
    body = client.post(
        "/optimize",
        json={
            "objective": "max_sharpe",
            "constraints": {
                "caps": {"private_equity": 0.10},
                "floors": {"cash": 0.10},
            },
            **FAST,
        },
    ).json()

    assert body["weights"]["private_equity"] <= 0.10 + 1e-6
    assert body["weights"]["cash"] >= 0.10 - 1e-6


def test_a_group_cap_is_applied(client):
    body = client.post(
        "/optimize",
        json={"objective": "max_sharpe", "constraints": {"group_caps": {"growth": 0.30}}, **FAST},
    ).json()
    growth = body["weights"]["equity"] + body["weights"]["private_equity"]
    assert growth <= 0.30 + 1e-6


def test_an_unknown_group_is_rejected_by_name(client):
    response = client.post(
        "/optimize", json={"constraints": {"group_caps": {"crypto": 0.5}}, **FAST}
    )
    assert response.status_code == 422
    assert "crypto" in str(response.json())


def test_impossible_floors_are_rejected(client):
    response = client.post(
        "/optimize",
        json={"constraints": {"floors": {"cash": 0.6, "equity": 0.6}}, **FAST},
    )
    assert response.status_code == 422


def test_the_near_optimal_range_is_returned(client):
    body = client.post(
        "/optimize", json={"objective": "max_sharpe", "tolerance": 0.05, **FAST}
    ).json()

    assert body["near_optimal_count"] > 1
    for row in body["ranges"]:
        assert row["low"] <= row["best"] + 1e-9
        assert row["best"] <= row["high"] + 1e-9


def test_the_frontier_is_ordered(client):
    body = client.post("/frontier", json=FAST).json()
    returns = [p["expected_return"] for p in body["points"]]
    vols = [p["volatility"] for p in body["points"]]

    assert returns == sorted(returns)
    assert vols == sorted(vols)


# ---------------------------------------------------------------------------
# Mandates
# ---------------------------------------------------------------------------

def test_a_reachable_mandate_returns_ranked_allocations(client):
    body = client.post(
        "/mandate",
        json={"target_return": 0.03, "max_volatility": 0.12, "limit": 5, **FAST},
    ).json()

    assert body["feasible"]
    assert len(body["allocations"]) <= 5
    assert body["envelope"]
    for allocation in body["allocations"]:
        assert allocation["realised_return"] >= 0.03 - 1e-9
        assert allocation["volatility"] <= 0.12 + 1e-9


def test_an_unreachable_mandate_says_what_would_have_to_change(client):
    """'Impossible' is true and useless. The value is in the trade."""
    body = client.post(
        "/mandate", json={"target_return": 0.30, "max_volatility": 0.06, **FAST}
    ).json()

    assert not body["feasible"]
    assert body["relaxations"]
    assert any("->" in r["description"] for r in body["relaxations"])


def test_ranking_choice_is_respected(client):
    body = client.post(
        "/mandate",
        json={
            "target_return": 0.03,
            "max_volatility": 0.15,
            "rank_by": "realised_return",
            "limit": 10,
            **FAST,
        },
    ).json()

    returns = [a["realised_return"] for a in body["allocations"]]
    assert returns == sorted(returns, reverse=True)


def test_an_unknown_ranking_column_is_rejected(client):
    response = client.post(
        "/mandate", json={"target_return": 0.03, "rank_by": "vibes", **FAST}
    )
    assert response.status_code == 422


def test_a_mandate_with_no_requirements_is_rejected(client):
    response = client.post("/mandate", json=FAST)
    assert response.status_code == 422


def test_percentages_sent_instead_of_fractions_are_rejected(client):
    """6 instead of 0.06 would otherwise solve silently and find nothing."""
    assert client.post("/mandate", json={"target_return": 6.0, **FAST}).status_code == 422
    assert (
        client.post("/mandate", json={"max_volatility": 10.0, **FAST}).status_code == 422
    )


def test_never_recovered_survives_json_as_null(client):
    """Infinity is how 'never regained the peak' is encoded, and it is not
    valid JSON. It must become null rather than crashing the response."""
    body = client.post(
        "/mandate",
        json={"target_return": 0.02, "max_volatility": 0.30, "limit": 50, **FAST},
    ).json()

    for allocation in body["allocations"]:
        recovery = allocation["months_to_recover"]
        assert recovery is None or recovery >= 0


def test_the_sweep_finds_where_the_target_stops_being_reachable(client):
    body = client.post(
        "/mandate/sweep",
        json={
            "max_volatility": 0.08,
            "target_from": 0.02,
            "target_to": 0.12,
            "target_step": 0.02,
            **FAST,
        },
    ).json()

    feasible = [p["feasible"] for p in body["points"]]
    assert feasible[0]
    assert not feasible[-1]
    assert body["highest_reachable"] is not None


def test_a_sweep_without_a_budget_is_rejected(client):
    response = client.post(
        "/mandate/sweep", json={"target_return": 0.05, **FAST}
    )
    assert response.status_code == 422


def test_an_absurd_sweep_range_is_refused(client):
    """A fine step across a wide range would mean hundreds of solves."""
    response = client.post(
        "/mandate/sweep",
        json={
            "max_volatility": 0.10,
            "target_from": 0.01,
            "target_to": 0.50,
            "target_step": 0.001,
            **FAST,
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

def test_period_comparison_returns_every_regime(client):
    body = client.post("/periods/compare", json={"samples": 500}).json()

    assert len(body["by_period"]) >= 3
    assert body["stability"]
    assert body["consensus"]
    assert body["average_premium"] is not None


def test_the_cross_period_matrix_is_square(client):
    body = client.post("/periods/compare", json={"samples": 500}).json()
    matrix = body["cross_period"]

    assert len(matrix["chosen_for"]) == len(matrix["measured_in"])
    assert len(matrix["sharpe"]) == len(matrix["chosen_for"])
    for row in matrix["sharpe"]:
        assert len(row) == len(matrix["measured_in"])


def test_the_consensus_allocation_is_valid(client):
    body = client.post("/periods/compare", json={"samples": 500}).json()
    assert sum(body["consensus"].values()) == pytest.approx(1.0, abs=1e-6)


def test_a_mandate_across_periods_names_the_binding_one(client):
    body = client.post(
        "/mandate/across-periods",
        json={"target_return": 0.02, "max_volatility": 0.20, "samples": 500},
    ).json()

    assert body["periods"]
    assert body["qualified_per_period"]
    assert "binding period" in body["explanation"]


# ---------------------------------------------------------------------------
# The interactive documentation
# ---------------------------------------------------------------------------

_EXAMPLE_ENDPOINTS = [
    ("MeasureRequest", "/measure"),
    ("OptimizeRequest", "/optimize"),
    ("MandateRequest", "/mandate"),
    ("SweepRequest", "/mandate/sweep"),
    ("PeriodsRequest", "/periods/compare"),
    ("RobustMandateRequest", "/mandate/across-periods"),
]


@pytest.mark.parametrize("schema_name,path", _EXAMPLE_ENDPOINTS)
def test_the_documented_example_actually_works(client, schema_name, path):
    """The docs page pre-fills its request body from these examples. Without
    them it invents placeholders -- "start": "string", a 400% return target --
    and the first thing anyone tries fails with a confusing error.

    So the examples have to be real requests, and this asserts they stay that
    way rather than drifting as the schemas change.
    """
    spec = client.get("/openapi.json").json()
    examples = spec["components"]["schemas"][schema_name].get("examples")
    assert examples, f"{schema_name} has no example for the docs to pre-fill"

    response = client.post(path, json=examples[0])
    assert response.status_code == 200, response.json()


def test_a_malformed_date_names_the_field(client):
    """pandas would say 'Unknown datetime string format, unable to parse:
    string' -- true, and silent about which field or what a good value is."""
    response = client.post(
        "/measure", json={"weights": _weights(), "start": "string"}
    )
    assert response.status_code == 422

    detail = str(response.json())
    assert "start" in detail
    assert "2010-01-01" in detail


def test_a_partial_date_is_rejected(client):
    response = client.post(
        "/measure", json={"weights": _weights(), "start": "2010-1-1"}
    )
    assert response.status_code == 422


def test_an_empty_date_is_treated_as_omitted(client):
    """A frontend clearing a date input sends an empty string, which should
    mean 'no bound' rather than being an error."""
    response = client.post(
        "/measure", json={"weights": _weights(), "start": "", "end": ""}
    )
    assert response.status_code == 200
