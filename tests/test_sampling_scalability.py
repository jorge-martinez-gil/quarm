from experiments.run_sampling_scalability import bounded_sparse_surface

from quarm.attribution import shapley_values


def test_controlled_surface_is_complete_bounded_and_nonadditive():
    players = tuple(f"mechanism_{index:02d}" for index in range(1, 13))
    values = bounded_sparse_surface(players)
    assert len(values) == 4096
    assert all(0.0 < value < 1.0 for value in values.values())
    interaction_residual = (
        values[frozenset(players[:2])]
        - values[frozenset({players[0]})]
        - values[frozenset({players[1]})]
        + values[frozenset()]
    )
    assert abs(interaction_residual) > 1e-4
    allocation = shapley_values(values, players)
    total = values[frozenset(players)] - values[frozenset()]
    assert abs(sum(allocation.values()) - total) < 1e-12
