import numpy as np
import pandas as pd
import pytest

from quarm.attribution import (
    all_coalitions,
    attribute_quality_debt,
    concentration_radius,
    shapley_interactions,
    shapley_values,
)


def test_additive_game_is_recovered_exactly():
    players = ["a", "b", "c"]
    weights = {"a": 0.1, "b": 0.25, "c": -0.04}
    values = {coalition: sum(weights[p] for p in coalition) for coalition in all_coalitions(players)}
    assert shapley_values(values, players) == pytest.approx(weights)
    assert all(abs(v) < 1e-12 for v in shapley_interactions(values, players).values())


def test_interaction_game_allocates_debt_and_exposes_synergy():
    players = ["a", "b"]
    values = {
        frozenset(): 0.2,
        frozenset({"a"}): 0.3,
        frozenset({"b"}): 0.4,
        frozenset({"a", "b"}): 0.8,
    }
    phi = shapley_values(values, players)
    assert phi["a"] == pytest.approx(0.25)
    assert phi["b"] == pytest.approx(0.35)
    assert sum(phi.values()) == pytest.approx(0.6)
    assert shapley_interactions(values, players)[("a", "b")] == pytest.approx(0.3)


def test_paired_bootstrap_is_deterministic_and_efficient():
    players = ["a", "b"]
    rows = []
    rng = np.random.default_rng(11)
    for repeat in range(10):
        shared = rng.normal(0, 0.01)
        for coalition in all_coalitions(players):
            rows.append(
                {
                    "repeat": repeat,
                    "coalition": coalition,
                    "loss": 0.2 + shared + 0.1 * ("a" in coalition) + 0.2 * ("b" in coalition),
                }
            )
    frame = pd.DataFrame(rows)
    first = attribute_quality_debt(frame, players, bootstrap_samples=300, seed=77)
    second = attribute_quality_debt(frame, players, bootstrap_samples=300, seed=77)
    pd.testing.assert_frame_equal(first.estimates, second.estimates)
    assert first.efficiency_residual == pytest.approx(0.0, abs=1e-12)


def test_concentration_radius_validation():
    assert 0 < concentration_radius(4, 1000) < 1
    with pytest.raises(ValueError):
        concentration_radius(0, 10)

