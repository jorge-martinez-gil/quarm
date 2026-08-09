"""Paired full-factorial estimation of a data-quality risk surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .attribution import all_coalitions, coalition_key
from .corruptions import Corruption, apply_corruptions

ModelFactory = Callable[[int], BaseEstimator]


@dataclass(frozen=True)
class StudyResult:
    observations: pd.DataFrame
    channels: tuple[str, ...]
    task: str
    severity: float
    repeats: int
    master_seed: int


def bounded_loss(y_true: NDArray, y_pred: NDArray, task: str) -> float:
    """A bounded loss suitable for cross-task comparison and concentration bounds."""
    if task == "classification":
        return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))
    if task == "regression":
        truth = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        scale = float(np.subtract(*np.quantile(truth, [0.75, 0.25])))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(truth))
        scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
        return float(np.mean(np.minimum(np.abs(pred - truth) / scale, 1.0)))
    raise ValueError("unknown task")


def _splitter(task: str, repeats: int, test_size: float, seed: int):
    if task == "classification":
        return StratifiedShuffleSplit(n_splits=repeats, test_size=test_size, random_state=seed)
    return ShuffleSplit(n_splits=repeats, test_size=test_size, random_state=seed)


def evaluate_risk_surface(
    X: NDArray,
    y: NDArray,
    channels: list[Corruption] | tuple[Corruption, ...],
    model_factory: ModelFactory,
    *,
    task: str,
    severity: float = 0.15,
    repeats: int = 12,
    test_size: float = 0.30,
    master_seed: int = 20260807,
) -> StudyResult:
    """Estimate risk for all channel coalitions using a paired experimental design."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-D and aligned with a 1-D y")
    if len(X) < 40:
        raise ValueError("at least 40 rows are required for a meaningful split")
    if len(channels) < 1:
        raise ValueError("at least one corruption channel is required")
    if len(channels) > 10:
        raise ValueError("exact evaluation is limited to 10 channels (2^m coalitions)")
    if repeats < 2 or not 0 < test_size < 1:
        raise ValueError("require repeats >= 2 and test_size in (0,1)")
    names = tuple(channel.name for channel in channels)
    if len(names) != len(set(names)):
        raise ValueError("channel names must be unique")
    lookup = {channel.name: channel for channel in channels}

    splitter = _splitter(task, repeats, test_size, master_seed)
    split_iter = splitter.split(X, y if task == "classification" else None)
    rows: list[dict] = []
    for repeat, (train_idx, test_idx) in enumerate(split_iter):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        for coalition in all_coalitions(names):
            selected = [lookup[name] for name in names if name in coalition]
            corrupted_X, corrupted_y, diagnostics = apply_corruptions(
                X_train, y_train, selected, severity, master_seed, repeat, task
            )
            model_seed = master_seed + repeat * 104729
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                model_factory(model_seed),
            )
            model.fit(corrupted_X, corrupted_y)
            prediction = model.predict(X_test)
            rows.append(
                {
                    "repeat": repeat,
                    "coalition": coalition,
                    "coalition_key": coalition_key(coalition),
                    "coalition_size": len(coalition),
                    "loss": bounded_loss(y_test, prediction, task),
                    "train_rows": len(corrupted_y),
                    "diagnostics": diagnostics,
                }
            )
    frame = pd.DataFrame(rows)
    expected = repeats * (2 ** len(names))
    if len(frame) != expected:
        raise RuntimeError(f"incomplete factorial design: {len(frame)} != {expected}")
    return StudyResult(frame, names, task, severity, repeats, master_seed)
