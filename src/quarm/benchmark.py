"""Pinned, offline benchmark registry and transparent model factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass(frozen=True)
class Dataset:
    name: str
    X: NDArray[np.float64]
    y: NDArray
    task: str
    provenance: str


def datasets() -> dict[str, Dataset]:
    cancer = load_breast_cancer()
    wine = load_wine()
    diabetes = load_diabetes()
    X_syn, y_syn = make_classification(
        n_samples=900,
        n_features=18,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=3,
        weights=[0.72, 0.28],
        class_sep=0.85,
        flip_y=0.01,
        random_state=7341,
    )
    return {
        "breast_cancer": Dataset(
            "breast_cancer",
            cancer.data.astype(float),
            cancer.target,
            "classification",
            "UCI Wisconsin Diagnostic Breast Cancer via scikit-learn 1.9.0",
        ),
        "wine": Dataset(
            "wine",
            wine.data.astype(float),
            wine.target,
            "classification",
            "UCI Wine Recognition via scikit-learn 1.9.0",
        ),
        "diabetes": Dataset(
            "diabetes",
            diabetes.data.astype(float),
            diabetes.target.astype(float),
            "regression",
            "Diabetes progression dataset via scikit-learn 1.9.0",
        ),
        "synthetic_nonlinear": Dataset(
            "synthetic_nonlinear",
            X_syn.astype(float),
            y_syn,
            "classification",
            "Pinned make_classification generator (seed 7341)",
        ),
    }


def model_factories(task: str) -> dict[str, Callable]:
    if task == "classification":
        return {
            "logistic": lambda seed: LogisticRegression(max_iter=3000, C=1.0, random_state=seed),
            "forest": lambda seed: RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=seed,
                n_jobs=1,
            ),
        }
    if task == "regression":
        return {
            "ridge": lambda seed: Ridge(alpha=1.0),
            "forest": lambda seed: RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=seed,
                n_jobs=1,
            ),
        }
    raise ValueError("unknown task")

