import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from quarm.corruptions import MissingCells, TargetNoise
from quarm.evaluation import bounded_loss, evaluate_risk_surface


def _data():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(160, 5))
    y = (X[:, 0] - 0.6 * X[:, 1] + rng.normal(0, 0.2, 160) > 0).astype(int)
    return X, y


def _factory(seed):
    return LogisticRegression(max_iter=1000, random_state=seed)


def test_full_factorial_shape_and_clean_test_pairing():
    X, y = _data()
    result = evaluate_risk_surface(
        X,
        y,
        [MissingCells(), TargetNoise()],
        _factory,
        task="classification",
        repeats=3,
        severity=0.1,
        master_seed=42,
    )
    assert len(result.observations) == 3 * 4
    assert result.observations.groupby("repeat").size().eq(4).all()
    assert set(result.observations["coalition_size"]) == {0, 1, 2}


def test_evaluation_is_reproducible():
    X, y = _data()
    kwargs = dict(
        channels=[MissingCells(), TargetNoise()],
        model_factory=_factory,
        task="classification",
        repeats=3,
        severity=0.1,
        master_seed=91,
    )
    first = evaluate_risk_surface(X, y, **kwargs)
    second = evaluate_risk_surface(X, y, **kwargs)
    pd.testing.assert_frame_equal(first.observations, second.observations)


def test_bounded_losses():
    assert bounded_loss(np.array([0, 1]), np.array([0, 0]), "classification") == 0.5
    loss = bounded_loss(np.array([0.0, 1.0, 2.0]), np.array([100.0, 1.0, 2.0]), "regression")
    assert 0 <= loss <= 1

