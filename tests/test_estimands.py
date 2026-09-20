"""Tests binding the manuscript's estimand claims to executable code."""

import numpy as np
import pandas as pd
import pytest

from quarm.attribution import (
    all_coalitions,
    attribute_quality_debt,
    banzhaf_weight_matrix,
    shapley_values,
    shapley_weight_matrix,
)
from quarm.relational import analytical_workload, normalize_query_weights
from quarm.sampling import banzhaf_values, estimate_permutation_shapley


def _paired_frame(values, repeats=6, jitter=0.01, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        shared = rng.normal(0, jitter)
        for coalition, value in values.items():
            rows.append({"repeat": repeat, "coalition": coalition, "loss": value + shared})
    return pd.DataFrame(rows)


def paper_counterexample():
    """The monotone bounded surface from the manuscript's Proposition 1 remark."""
    return {
        frozenset(): 0.0,
        frozenset({"A"}): 0.4,
        frozenset({"B"}): 0.1,
        frozenset({"C"}): 0.1,
        frozenset({"A", "B"}): 0.6,
        frozenset({"A", "C"}): 0.4,
        frozenset({"B", "C"}): 0.5,
        frozenset({"A", "B", "C"}): 0.8,
    }


def test_paper_counterexample_ranks_accounting_and_intervention_differently():
    values = paper_counterexample()
    players = ("A", "B", "C")
    phi = shapley_values(values, players)
    assert phi["A"] == pytest.approx(0.3667, abs=1e-4)
    assert phi["B"] == pytest.approx(0.2667, abs=1e-4)
    assert phi["C"] == pytest.approx(0.1667, abs=1e-4)
    full = frozenset(players)
    gains = {p: values[full] - values[full - {p}] for p in players}
    assert gains == pytest.approx({"A": 0.3, "B": 0.4, "C": 0.2})
    assert max(phi, key=phi.get) == "A"
    assert max(gains, key=gains.get) == "B"


def test_attribution_reports_all_three_estimands_and_they_agree_when_additive():
    players = ("a", "b", "c")
    weights = {"a": 0.12, "b": 0.05, "c": -0.02}
    values = {c: sum(weights[p] for p in c) for c in all_coalitions(players)}
    frame = _paired_frame(values)
    result = attribute_quality_debt(frame, players, bootstrap_samples=200, seed=5)
    for player in players:
        assert result.singletons[player] == pytest.approx(weights[player], abs=1e-9)
        assert result.gains[player] == pytest.approx(weights[player], abs=1e-9)
        assert result.banzhaf[player] == pytest.approx(weights[player], abs=1e-9)
    assert result.banzhaf_residual == pytest.approx(0.0, abs=1e-9)
    assert {"singleton", "current_gain", "gain_ci_low", "gain_ci_high", "banzhaf"} <= set(
        result.estimates.columns
    )


def test_attribution_estimands_disagree_on_the_counterexample():
    values = paper_counterexample()
    players = ("A", "B", "C")
    frame = _paired_frame(values, jitter=0.0)
    result = attribute_quality_debt(frame, players, bootstrap_samples=200, seed=1)
    top_debt = result.estimates.iloc[0]["channel"]
    top_gain = max(result.gains, key=result.gains.get)
    assert top_debt == "A"
    assert top_gain == "B"
    assert result.efficiency_residual == pytest.approx(0.0, abs=1e-12)


def test_weight_matrices_match_definitional_computations():
    rng = np.random.default_rng(17)
    players = tuple("pqrst")
    columns = list(all_coalitions(players))
    values = {c: float(rng.uniform(0, 1)) for c in columns}
    vector = np.array([values[c] for c in columns])
    shapley_direct = shapley_values(values, players)
    shapley_fast = shapley_weight_matrix(players, columns) @ vector
    for idx, player in enumerate(players):
        assert shapley_fast[idx] == pytest.approx(shapley_direct[player], abs=1e-12)
    banzhaf_direct = banzhaf_values(values, players)
    banzhaf_fast = banzhaf_weight_matrix(players, columns) @ vector
    for idx, player in enumerate(players):
        assert banzhaf_fast[idx] == pytest.approx(banzhaf_direct[player], abs=1e-12)


def test_antithetic_sampling_is_exact_for_additive_games_and_converges():
    players = ["a", "b", "c", "d"]
    weights = {"a": 0.02, "b": -0.03, "c": 0.11, "d": 0.07}

    def additive(coalition):
        return sum(weights[p] for p in coalition)

    result = estimate_permutation_shapley(additive, players, permutations=8, seed=2, antithetic=True)
    assert result.values == pytest.approx(weights)
    assert result.antithetic

    def interacting(coalition):
        return 0.1 * ("a" in coalition) + 0.2 * ("b" in coalition) + 0.3 * ({"a", "c"} <= coalition)

    exact = shapley_values(
        {c: interacting(c) for c in all_coalitions(players)}, players
    )
    sampled = estimate_permutation_shapley(
        interacting, players, permutations=2_000, seed=9, antithetic=True
    )
    assert sampled.values == pytest.approx(exact, abs=0.01)


def test_antithetic_sampling_rejects_odd_budgets_and_bad_delta():
    with pytest.raises(ValueError):
        estimate_permutation_shapley(lambda c: 0.0, ["a", "b"], permutations=3, antithetic=True)
    with pytest.raises(ValueError):
        estimate_permutation_shapley(lambda c: 0.0, ["a", "b"], permutations=4, delta=1.5)


def test_query_weight_policy_is_validated_and_normalized():
    workload = analytical_workload()
    equal = normalize_query_weights(workload, None)
    assert sum(equal.values()) == pytest.approx(1.0)
    assert len(set(equal.values())) == 1
    custom = normalize_query_weights(workload, {"revenue_region": 3.0, "daily_revenue": 1.0})
    assert custom["revenue_region"] == pytest.approx(0.75)
    assert custom["daily_revenue"] == pytest.approx(0.25)
    assert custom["aov_segment"] == 0.0
    with pytest.raises(ValueError):
        normalize_query_weights(workload, {"no_such_query": 1.0})
    with pytest.raises(ValueError):
        normalize_query_weights(workload, {"revenue_region": -1.0})
