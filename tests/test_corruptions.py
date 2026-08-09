import numpy as np
import pytest

from quarm.corruptions import (
    DuplicateRows,
    FeatureNoise,
    MissingCells,
    TargetNoise,
    apply_corruptions,
)


@pytest.fixture
def data():
    X = np.arange(120, dtype=float).reshape(30, 4)
    y = np.tile([0, 1], 15)
    return X, y


def test_missingness_has_exact_count(data):
    X, y = data
    out, same_y, info = MissingCells().apply(X, y, 0.10, np.random.default_rng(7), "classification")
    assert np.isnan(out).sum() == 12
    assert np.array_equal(same_y, y)
    assert info["affected_cells"] == 12


def test_channels_do_not_mutate_inputs(data):
    X, y = data
    original_X, original_y = X.copy(), y.copy()
    for channel in [MissingCells(), FeatureNoise(), TargetNoise(), DuplicateRows()]:
        channel.apply(X, y, 0.2, np.random.default_rng(3), "classification")
    assert np.array_equal(X, original_X)
    assert np.array_equal(y, original_y)


def test_channel_randomness_is_subset_invariant(data):
    X, y = data
    missing_only, _, _ = apply_corruptions(
        X, y, [MissingCells()], 0.2, master_seed=19, repeat=2, task="classification"
    )
    missing_and_target, _, _ = apply_corruptions(
        X,
        y,
        [MissingCells(), TargetNoise()],
        0.2,
        master_seed=19,
        repeat=2,
        task="classification",
    )
    assert np.array_equal(np.isnan(missing_only), np.isnan(missing_and_target))


def test_duplicate_channel_is_last_and_preserves_alignment(data):
    X, y = data
    out_X, out_y, diagnostics = apply_corruptions(
        X,
        y,
        [DuplicateRows(), MissingCells()],
        0.2,
        master_seed=5,
        repeat=0,
        task="classification",
    )
    assert len(out_X) == len(out_y) == 36
    assert diagnostics["duplicate_rows"]["added_rows"] == 6


@pytest.mark.parametrize("severity", [-0.1, 1.1])
def test_invalid_severity_is_rejected(data, severity):
    X, y = data
    with pytest.raises(ValueError):
        MissingCells().apply(X, y, severity, np.random.default_rng(0), "classification")

