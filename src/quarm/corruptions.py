"""Auditable data-defect channels with deterministic composition semantics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Target = NDArray[np.float64] | NDArray[np.int64]


class Corruption(Protocol):
    """Protocol for a parameterized training-data defect channel."""

    name: str
    priority: int

    def apply(
        self,
        X: Array,
        y: Target,
        severity: float,
        rng: np.random.Generator,
        task: str,
    ) -> tuple[Array, Target, dict[str, float | int | str]]: ...


def _validate(X: Array, y: Target, severity: float, task: str) -> None:
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-D and y must be aligned and 1-D")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")


def _count(rate: float, population: int) -> int:
    """Return a deterministic count, preserving zero at zero severity."""
    if rate == 0 or population == 0:
        return 0
    return min(population, max(1, int(round(rate * population))))


@dataclass(frozen=True)
class MissingCells:
    """Replace a fixed fraction of feature cells by NaN (MCAR stress channel)."""

    name: str = "missing_cells"
    priority: int = 10

    def apply(self, X: Array, y: Target, severity: float, rng: np.random.Generator, task: str):
        _validate(X, y, severity, task)
        out = np.array(X, dtype=float, copy=True)
        n = _count(severity, out.size)
        if n:
            idx = rng.choice(out.size, size=n, replace=False)
            out.flat[idx] = np.nan
        return out, y.copy(), {"affected_cells": n, "realized_rate": n / max(1, out.size)}


@dataclass(frozen=True)
class FeatureNoise:
    """Add scale-aware Gaussian noise to a fixed fraction of feature cells."""

    magnitude: float = 3.0
    name: str = "feature_noise"
    priority: int = 20

    def apply(self, X: Array, y: Target, severity: float, rng: np.random.Generator, task: str):
        _validate(X, y, severity, task)
        out = np.array(X, dtype=float, copy=True)
        n = _count(severity, out.size)
        if n:
            idx = rng.choice(out.size, size=n, replace=False)
            rows, cols = np.unravel_index(idx, out.shape)
            scales = np.nanstd(out, axis=0)
            scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
            out[rows, cols] += rng.normal(0.0, self.magnitude * scales[cols], size=n)
        return out, y.copy(), {
            "affected_cells": n,
            "realized_rate": n / max(1, out.size),
            "magnitude_sd": self.magnitude,
        }


@dataclass(frozen=True)
class TargetNoise:
    """Flip class labels or perturb regression targets for selected rows."""

    magnitude: float = 2.0
    name: str = "target_noise"
    priority: int = 30

    def apply(self, X: Array, y: Target, severity: float, rng: np.random.Generator, task: str):
        _validate(X, y, severity, task)
        target = np.array(y, copy=True)
        n = _count(severity, len(target))
        if n:
            rows = rng.choice(len(target), size=n, replace=False)
            if task == "classification":
                classes = np.unique(target)
                if len(classes) < 2:
                    raise ValueError("target noise requires at least two classes")
                for row in rows:
                    alternatives = classes[classes != target[row]]
                    target[row] = rng.choice(alternatives)
            else:
                scale = float(np.nanstd(target.astype(float)))
                scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
                target = target.astype(float)
                target[rows] += rng.normal(0.0, self.magnitude * scale, size=n)
        return X.copy(), target, {
            "affected_rows": n,
            "realized_rate": n / max(1, len(target)),
            "magnitude_sd": self.magnitude if task == "regression" else "class_flip",
        }


@dataclass(frozen=True)
class DuplicateRows:
    """Append exact copies of sampled rows, modeling multiplicity errors."""

    name: str = "duplicate_rows"
    priority: int = 100

    def apply(self, X: Array, y: Target, severity: float, rng: np.random.Generator, task: str):
        _validate(X, y, severity, task)
        n = _count(severity, len(y))
        if not n:
            return X.copy(), y.copy(), {"added_rows": 0, "realized_rate": 0.0}
        rows = rng.choice(len(y), size=n, replace=True)
        return (
            np.concatenate([X, X[rows]], axis=0),
            np.concatenate([y, y[rows]], axis=0),
            {"added_rows": n, "realized_rate": n / max(1, len(y))},
        )


def default_corruptions() -> list[Corruption]:
    return [MissingCells(), FeatureNoise(), TargetNoise(), DuplicateRows()]


def channel_seed(master_seed: int, repeat: int, channel_name: str) -> int:
    """Stable cross-process seed; unlike Python's hash, this is not randomized."""
    payload = f"quarm|{master_seed}|{repeat}|{channel_name}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "little", signed=False)


def apply_corruptions(
    X: Array,
    y: Target,
    channels: list[Corruption] | tuple[Corruption, ...],
    severity: float,
    master_seed: int,
    repeat: int,
    task: str,
) -> tuple[Array, Target, dict[str, dict[str, float | int | str]]]:
    """Compose channels deterministically with subset-invariant random streams."""
    names = [c.name for c in channels]
    if len(names) != len(set(names)):
        raise ValueError("corruption channel names must be unique")
    out_X, out_y = np.asarray(X, dtype=float).copy(), np.asarray(y).copy()
    diagnostics: dict[str, dict[str, float | int | str]] = {}
    for channel in sorted(channels, key=lambda c: (c.priority, c.name)):
        rng = np.random.default_rng(channel_seed(master_seed, repeat, channel.name))
        out_X, out_y, info = channel.apply(out_X, out_y, severity, rng, task)
        diagnostics[channel.name] = info
    return out_X, out_y, diagnostics

