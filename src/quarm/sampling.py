"""Scalable permutation sampling for mechanism-level Shapley debt."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from typing import Callable, Mapping

import numpy as np

from .attribution import Coalition, _check_values


@dataclass(frozen=True)
class PermutationEstimate:
    """Sampled Shapley debts with everything needed to audit the estimate.

    `contributions` retains the raw marginal matrix (one row per sampled
    order), `unique_coalitions` records how much of the exact surface the
    cache actually touched, and `hoeffding_radius` is the conservative
    simultaneous finite-sample radius at the requested failure probability.
    """

    values: dict[str, float]
    standard_errors: dict[str, float]
    samples: int
    unique_coalitions: int
    exact_coalitions: int
    hoeffding_radius: float
    contributions: np.ndarray
    antithetic: bool = field(default=False)


def banzhaf_values(values: Mapping[Coalition, float], players: list[str] | tuple[str, ...]) -> dict[str, float]:
    """Exact normalized Banzhaf values for a complete mechanism game."""
    players = tuple(players)
    _check_values(values, players)
    if not players:
        return {}
    weight = 1.0 / (2 ** (len(players) - 1))
    result = {}
    for player in players:
        others = [candidate for candidate in players if candidate != player]
        total = 0.0
        for mask in range(2 ** len(others)):
            coalition = frozenset(others[idx] for idx in range(len(others)) if mask & (1 << idx))
            total += values[coalition | {player}] - values[coalition]
        result[player] = float(weight * total)
    return result


def estimate_permutation_shapley(
    value: Callable[[Coalition], float],
    players: list[str] | tuple[str, ...],
    *,
    permutations: int,
    seed: int = 0,
    loss_range: float = 1.0,
    antithetic: bool = False,
    delta: float = 0.05,
) -> PermutationEstimate:
    """Estimate Shapley debt from random repair orders with coalition caching.

    `value` may execute an expensive workload. Only prefix coalitions visited by
    sampled permutations are evaluated, and repeated prefixes are cached.

    With `antithetic=True`, orders are drawn in reversed pairs: each random
    order is followed by its mirror image. Every player then sees each sampled
    order once from the front and once from the back, which cancels the
    additive component of the game exactly and often reduces variance on
    near-additive surfaces at zero extra evaluation cost (mirror prefixes are
    complements, so caching still applies). Standard errors are computed over
    independent pair means so the reported uncertainty stays valid.
    """
    players = tuple(players)
    if len(players) < 1 or permutations < 2 or loss_range <= 0:
        raise ValueError("require players, permutations >= 2, and loss_range > 0")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    if antithetic and permutations % 2:
        raise ValueError("antithetic sampling requires an even permutation count")
    rng = np.random.default_rng(seed)
    cache: dict[Coalition, float] = {}

    def cached(coalition: Coalition) -> float:
        if coalition not in cache:
            cache[coalition] = float(value(coalition))
        return cache[coalition]

    if antithetic:
        base = [list(rng.permutation(players)) for _ in range(permutations // 2)]
        orders = [order for draw in base for order in (draw, draw[::-1])]
    else:
        orders = [list(rng.permutation(players)) for _ in range(permutations)]

    samples = np.empty((permutations, len(players)), dtype=float)
    player_index = {player: idx for idx, player in enumerate(players)}
    for draw, order in enumerate(orders):
        coalition: Coalition = frozenset()
        previous = cached(coalition)
        for player in order:
            expanded = coalition | {player}
            current = cached(expanded)
            samples[draw, player_index[player]] = current - previous
            coalition, previous = expanded, current
    estimates = samples.mean(axis=0)
    if antithetic:
        pair_means = samples.reshape(permutations // 2, 2, len(players)).mean(axis=1)
        errors = pair_means.std(axis=0, ddof=1) / sqrt(permutations // 2)
        independent = permutations // 2
    else:
        errors = samples.std(axis=0, ddof=1) / sqrt(permutations)
        independent = permutations
    # Each marginal lies in [-loss_range, loss_range], hence interval length 2*range.
    radius = loss_range * sqrt(2.0 * log(2.0 * len(players) / delta) / independent)
    return PermutationEstimate(
        dict(zip(players, estimates, strict=True)),
        dict(zip(players, errors, strict=True)),
        permutations,
        len(cache),
        2 ** len(players),
        float(radius),
        samples,
        antithetic,
    )
