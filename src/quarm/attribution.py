"""Exact Shapley quality-debt attribution, interactions, and uncertainty."""

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
    estimates: pd.DataFrame
    interactions: pd.DataFrame
    total_debt: float
    efficiency_residual: float
    coalition_means: dict[Coalition, float]


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

    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_samples, len(players)), dtype=float)
    matrix = pivot.to_numpy()
    columns = list(pivot.columns)
    for b in range(bootstrap_samples):
        rows = rng.integers(0, len(repeats), size=len(repeats))
        sampled = matrix[rows].mean(axis=0)
        values = dict(zip(columns, sampled, strict=True))
        draw = shapley_values(values, players)
        boot[b] = [draw[p] for p in players]

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
            }
        )
    estimates = pd.DataFrame(records).sort_values("debt", ascending=False, ignore_index=True)
    interaction_frame = pd.DataFrame(
        [{"left": a, "right": b, "interaction": v} for (a, b), v in interactions.items()]
    ).sort_values("interaction", ascending=False, ignore_index=True)
    total_debt = mean_values[frozenset(players)] - mean_values[frozenset()]
    residual = sum(point.values()) - total_debt
    return AttributionResult(estimates, interaction_frame, total_debt, float(residual), mean_values)

