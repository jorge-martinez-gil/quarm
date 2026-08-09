import pytest

from quarm.attribution import all_coalitions, shapley_values
from quarm.sampling import banzhaf_values, estimate_permutation_shapley


def test_permutation_estimator_is_exact_for_additive_game():
    players = ["a", "b", "c", "d"]
    weights = {"a": 0.02, "b": -0.03, "c": 0.11, "d": 0.07}

    def value(coalition):
        return sum(weights[player] for player in coalition)

    result = estimate_permutation_shapley(value, players, permutations=20, seed=8)
    assert result.values == pytest.approx(weights)
    assert result.unique_coalitions <= result.exact_coalitions
    assert all(error < 1e-16 for error in result.standard_errors.values())


def test_sampled_estimate_converges_on_interacting_game():
    players = ["a", "b", "c"]

    def value(coalition):
        return 0.1 * ("a" in coalition) + 0.2 * ("b" in coalition) + 0.3 * ({"a", "c"} <= coalition)

    table = {coalition: value(coalition) for coalition in all_coalitions(players)}
    exact = shapley_values(table, players)
    sampled = estimate_permutation_shapley(value, players, permutations=2_000, seed=5)
    assert sampled.values == pytest.approx(exact, abs=0.01)


def test_banzhaf_does_not_force_exact_accounting():
    players = ["a", "b", "c"]
    table = {coalition: float(len(coalition) == 3) for coalition in all_coalitions(players)}
    shapley = shapley_values(table, players)
    banzhaf = banzhaf_values(table, players)
    assert sum(shapley.values()) == pytest.approx(1.0)
    assert sum(banzhaf.values()) == pytest.approx(0.75)
