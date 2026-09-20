"""Exact Shapley quality-debt attribution, interactions, and uncertainty.

This module computes every estimand the manuscript reports from one complete
coalition table: exact Shapley debts, pairwise Shapley interactions, singleton
stress effects, current-state repair gains, and normalized Banzhaf comparators.
The bootstrap is paired at the repeat level and vectorized: each exact Shapley
allocation is a single matrix product against a precomputed weight matrix, so
thousands of bootstrap draws cost about as much as one Python-loop allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial, log, sqrt
from typing import Mapping

import numpy as np
import pandas as pd


Coalition = frozenset[str]


@dataclass(frozen=True)
class AttributionResult:
    """Every estimand derived from one paired coalition study.

    `estimates` carries, per mechanism, the Shapley debt with paired-bootstrap
    intervals plus the singleton effect, the current-state repair gain with its
    own paired interval, and the normalized Banzhaf comparator. The scalar
    fields preserve the accounting audit trail: `efficiency_residual` must be
    zero to floating-point precision, while `banzhaf_residual` is reported
    rather than renormalized because Banzhaf is not an efficient allocation.
    """

    estimates: pd.DataFrame
    interactions: pd.DataFrame
    total_debt: float
    efficiency_residual: float
    coalition_means: dict[Coalition, float]
    singletons: dict[str, float]
    gains: dict[str, float]
    banzhaf: dict[str, float]
    banzhaf_residual: float


def coalition_key(members: Coalition | set[str] | tuple[str, ...]) -> str:
    return "|".join(sorted(members)) if members else "EMPTY"


def all_coalitions(players: list[str] | tuple[str, ...]):
    ordered = tuple(players)
    for size in range(len(ordered) + 1):
        for subset in combinations(ordered, size):
            yield frozenset(subset)


def _check_values(values: Mapping[Coalition, float], players: tuple[str, ...]) -> None:
    expected = set(all_coalitions(players))
    missing = expected.difference(values)
    extra = set(values).difference(expected)
    if missing or extra:
        raise ValueError(
            f"coalition map mismatch; missing={sorted(map(coalition_key, missing))}, "
            f"extra={sorted(map(coalition_key, extra))}"
        )


def shapley_values(values: Mapping[Coalition, float], players: list[str] | tuple[str, ...]) -> dict[str, float]:
    """Compute the exact Shapley allocation for a complete coalition table."""
    players = tuple(players)
    _check_values(values, players)
    m = len(players)
    if m == 0:
        return {}
    denom = factorial(m)
    result: dict[str, float] = {}
    for player in players:
        others = [p for p in players if p != player]
        total = 0.0
        for subset in all_coalitions(others):
            weight = factorial(len(subset)) * factorial(m - len(subset) - 1) / denom
            total += weight * (values[subset | {player}] - values[subset])
        result[player] = float(total)
    return result


def shapley_interactions(
    values: Mapping[Coalition, float], players: list[str] | tuple[str, ...]
) -> dict[tuple[str, str], float]:
    """Compute the pairwise Shapley interaction index.

    Positive values mean the two channels are super-additive in risk; negative
    values mean their joint harm is less than the sum of contextual marginals.
    """
    players = tuple(players)
    _check_values(values, players)
    m = len(players)
    if m < 2:
        return {}
    result: dict[tuple[str, str], float] = {}
    for i, left in enumerate(players):
        for right in players[i + 1 :]:
            others = [p for p in players if p not in {left, right}]
            value = 0.0
            for subset in all_coalitions(others):
                weight = factorial(len(subset)) * factorial(m - len(subset) - 2) / factorial(m - 1)
                second = (
                    values[subset | {left, right}]
                    - values[subset | {left}]
                    - values[subset | {right}]
                    + values[subset]
                )
                value += weight * second
            result[(left, right)] = float(value)
    return result


def shapley_weight_matrix(players: tuple[str, ...], columns: list[Coalition]) -> np.ndarray:
    """Return the linear map from coalition values to exact Shapley values.

    Row i holds the signed factorial weights so that `matrix @ values` equals
    the Shapley allocation for any complete game laid out in `columns` order.
    The map exists because the Shapley value is linear in the game; it lets a
    paired bootstrap recompute thousands of allocations as one matrix product.
    """
    m = len(players)
    denom = factorial(m)
    index = {player: position for position, player in enumerate(players)}
    matrix = np.zeros((m, len(columns)), dtype=float)
    for column, coalition in enumerate(columns):
        size = len(coalition)
        inside = factorial(size - 1) * factorial(m - size) / denom if size else 0.0
        outside = factorial(size) * factorial(m - size - 1) / denom if size < m else 0.0
        for player in players:
            if player in coalition:
                matrix[index[player], column] = inside
            else:
                matrix[index[player], column] = -outside
    return matrix


def banzhaf_weight_matrix(players: tuple[str, ...], columns: list[Coalition]) -> np.ndarray:
    """Return the linear map from coalition values to normalized Banzhaf values."""
    m = len(players)
    weight = 1.0 / (2 ** (m - 1)) if m else 0.0
    index = {player: position for position, player in enumerate(players)}
    matrix = np.zeros((m, len(columns)), dtype=float)
    for column, coalition in enumerate(columns):
        for player in players:
            matrix[index[player], column] = weight if player in coalition else -weight
    return matrix


def concentration_radius(n_channels: int, repeats: int, delta: float = 0.05) -> float:
    """Conservative simultaneous Hoeffding radius for bounded [0,1] loss debt."""
    if n_channels < 1 or repeats < 1 or not 0 < delta < 1:
        raise ValueError("require n_channels >= 1, repeats >= 1, and delta in (0,1)")
    epsilon = sqrt(log((2 ** (n_channels + 1)) / delta) / (2 * repeats))
    return min(2.0, 2.0 * epsilon)


def attribute_quality_debt(
    observations: pd.DataFrame,
    players: list[str] | tuple[str, ...],
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> AttributionResult:
    """Attribute risk using paired repeated coalition observations.

    `observations` must contain repeat, coalition (a `frozenset`), and loss.
    Bootstrap resampling occurs at the repeat level, preserving all pairing.
    The result separates three questions that a single ranking conflates:
    the Shapley debt (average-order accounting), the singleton effect
    (one-defect stress testing), and the current-state gain (removal from the
    fully defective joint state). Their disagreement certifies non-additivity.
    """
    required = {"repeat", "coalition", "loss"}
    if not required.issubset(observations.columns):
        raise ValueError(f"observations must contain {sorted(required)}")
    players = tuple(players)
    repeats = np.array(sorted(observations["repeat"].unique()))
    if len(repeats) < 2:
        raise ValueError("at least two paired repeats are required")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")

    pivot = observations.pivot(index="repeat", columns="coalition", values="loss").reindex(repeats)
    _check_values({c: 0.0 for c in pivot.columns}, players)
    if pivot.isna().any().any():
        raise ValueError("every repeat must contain every coalition exactly once")

    mean_values = {coalition: float(pivot[coalition].mean()) for coalition in pivot.columns}
    point = shapley_values(mean_values, players)
    interactions = shapley_interactions(mean_values, players)

    full = frozenset(players)
    empty: Coalition = frozenset()
    singletons = {p: mean_values[frozenset({p})] - mean_values[empty] for p in players}
    gains = {p: mean_values[full] - mean_values[full - {p}] for p in players}

    columns = list(pivot.columns)
    column_index = {coalition: position for position, coalition in enumerate(columns)}
    shapley_map = shapley_weight_matrix(players, columns)
    banzhaf_map = banzhaf_weight_matrix(players, columns)
    means_vector = np.array([mean_values[c] for c in columns])
    banzhaf_point = banzhaf_map @ means_vector
    banzhaf = dict(zip(players, map(float, banzhaf_point), strict=True))
    total_debt = mean_values[full] - mean_values[empty]
    banzhaf_residual = float(banzhaf_point.sum() - total_debt)

    rng = np.random.default_rng(seed)
    matrix = pivot.to_numpy()
    draws = np.empty((bootstrap_samples, len(repeats)), dtype=np.int64)
    for b in range(bootstrap_samples):
        draws[b] = rng.integers(0, len(repeats), size=len(repeats))
    sampled_means = matrix[draws].mean(axis=1)
    boot = sampled_means @ shapley_map.T
    full_column = column_index[full]
    gain_boot = np.column_stack(
        [sampled_means[:, full_column] - sampled_means[:, column_index[full - {p}]] for p in players]
    )

    alpha = (1.0 - confidence) / 2.0
    records = []
    for idx, player in enumerate(players):
        records.append(
            {
                "channel": player,
                "debt": point[player],
                "ci_low": float(np.quantile(boot[:, idx], alpha)),
                "ci_high": float(np.quantile(boot[:, idx], 1 - alpha)),
                "probability_harmful": float(np.mean(boot[:, idx] > 0)),
                "rank_1_probability": float(np.mean(np.argmax(boot, axis=1) == idx)),
                "singleton": singletons[player],
                "current_gain": gains[player],
                "gain_ci_low": float(np.quantile(gain_boot[:, idx], alpha)),
                "gain_ci_high": float(np.quantile(gain_boot[:, idx], 1 - alpha)),
                "banzhaf": banzhaf[player],
            }
        )
    estimates = pd.DataFrame(records).sort_values("debt", ascending=False, ignore_index=True)
    interaction_frame = pd.DataFrame(
        [{"left": a, "right": b, "interaction": v} for (a, b), v in interactions.items()]
    ).sort_values("interaction", ascending=False, ignore_index=True)
    residual = sum(point.values()) - total_debt
    return AttributionResult(
        estimates,
        interaction_frame,
        total_debt,
        float(residual),
        mean_values,
        singletons,
        gains,
        banzhaf,
        banzhaf_residual,
    )
