"""Controlled 12-mechanism study for permutation-Shapley scalability.

The exact 2^12 response surface is retained as ground truth.  The bounded
surface deliberately contains sparse positive and negative interactions so the
experiment measures approximation of a non-additive game, not recovery of a
trivial linear score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quarm.attribution import all_coalitions, coalition_key, shapley_values
from quarm.io import sha256_file, write_json
from quarm.sampling import estimate_permutation_shapley

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "icde_results"
FIGURES = ROOT / "artifacts" / "icde_figures"


def bounded_sparse_surface(players: tuple[str, ...]) -> dict[frozenset[str], float]:
    """Return a reproducible, bounded, non-additive mechanism-risk surface."""
    if len(players) != 12:
        raise ValueError("the controlled scalability surface requires 12 mechanisms")
    weights = dict(zip(players, [0.36, 0.29, 0.24, 0.19, 0.16, 0.13, 0.11, 0.09, 0.07, 0.055, 0.04, 0.025], strict=True))
    pairs = {
        (players[0], players[1]): -0.19,
        (players[0], players[5]): 0.13,
        (players[1], players[2]): 0.16,
        (players[2], players[7]): -0.12,
        (players[3], players[4]): 0.11,
        (players[4], players[8]): -0.08,
        (players[6], players[9]): 0.07,
        (players[8], players[10]): 0.05,
    }
    triples = {
        (players[0], players[1], players[2]): 0.14,
        (players[3], players[4], players[5]): -0.10,
        (players[7], players[8], players[9]): 0.08,
    }

    values: dict[frozenset[str], float] = {}
    for coalition in all_coalitions(players):
        latent = -1.9 + sum(weights[player] for player in coalition)
        latent += sum(effect for members, effect in pairs.items() if set(members) <= coalition)
        latent += sum(effect for members, effect in triples.items() if set(members) <= coalition)
        # Logistic mapping makes every response a valid loss in (0, 1).
        values[coalition] = float(1.0 / (1.0 + np.exp(-latent)))
    return values


def plot_scalability(frame: pd.DataFrame) -> list[Path]:
    grouped = frame.groupby("permutations").agg(
        rmse_median=("rmse", "median"),
        rmse_low=("rmse", lambda x: np.quantile(x, 0.10)),
        rmse_high=("rmse", lambda x: np.quantile(x, 0.90)),
        top1=("top1", "mean"),
        coalitions=("coalition_fraction", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    x = grouped["permutations"]
    axes[0].plot(x, 100 * grouped["rmse_median"], "o-", color="#0072B2", lw=1.7)
    axes[0].fill_between(x, 100 * grouped["rmse_low"], 100 * grouped["rmse_high"], color="#0072B2", alpha=0.18)
    axes[0].set_ylabel("Attribution RMSE (risk points)")
    axes[0].set_xlabel("Sampled permutations")
    axes[1].plot(x, 100 * grouped["top1"], "o-", color="#009E73", label="top-1 recovered")
    axes[1].plot(x, 100 * grouped["coalitions"], "s-", color="#D55E00", label="surface evaluated")
    axes[1].set_ylabel("Percent")
    axes[1].set_xlabel("Sampled permutations")
    axes[1].set_ylim(0, 105)
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.22)
    fig.tight_layout()
    pdf = FIGURES / "icde_figure4_sampling_m12.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20270807)
    args = parser.parse_args()
    if args.trials < 20:
        raise ValueError("use at least 20 trials")
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    players = tuple(f"mechanism_{index:02d}" for index in range(1, 13))
    surface = bounded_sparse_surface(players)
    exact = shapley_values(surface, players)
    exact_vector = np.array([exact[player] for player in players])
    exact_top = players[int(np.argmax(exact_vector))]
    rows = []
    for permutations in [4, 8, 16, 32, 64, 128, 256, 512]:
        for trial in range(args.trials):
            estimate = estimate_permutation_shapley(
                surface.__getitem__,
                players,
                permutations=permutations,
                seed=args.seed + 100_000 * permutations + trial,
            )
            estimate_vector = np.array([estimate.values[player] for player in players])
            rows.append(
                {
                    "permutations": permutations,
                    "trial": trial,
                    "rmse": float(np.sqrt(np.mean((estimate_vector - exact_vector) ** 2))),
                    "max_error": float(np.max(np.abs(estimate_vector - exact_vector))),
                    "top1": players[int(np.argmax(estimate_vector))] == exact_top,
                    "unique_coalitions": estimate.unique_coalitions,
                    "exact_coalitions": estimate.exact_coalitions,
                    "coalition_fraction": estimate.unique_coalitions / estimate.exact_coalitions,
                    "hoeffding_radius": estimate.hoeffding_radius,
                }
            )
    frame = pd.DataFrame(rows)
    results_path = RESULTS / "sampling_scalability_m12.csv"
    surface_path = RESULTS / "sampling_surface_m12.csv"
    exact_path = RESULTS / "sampling_exact_m12.csv"
    frame.to_csv(results_path, index=False)
    pd.DataFrame(
        [{"coalition": coalition_key(coalition), "loss": loss} for coalition, loss in surface.items()]
    ).to_csv(surface_path, index=False)
    pd.DataFrame([{"channel": player, "debt": exact[player]} for player in players]).to_csv(exact_path, index=False)
    figures = plot_scalability(frame)
    generated = [results_path, surface_path, exact_path, *figures]
    manifest = {
        "study": "QuaRM controlled 12-mechanism permutation scalability",
        "configuration": vars(args),
        "design": {
            "channels": 12,
            "exact_coalitions": 4096,
            "surface": "bounded logistic sparse polynomial with eight pair and three triple interactions",
            "ground_truth": "exact Shapley allocation over all coalitions",
        },
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in generated},
    }
    write_json(RESULTS / "sampling_scalability_m12_manifest.json", manifest)
    print(json.dumps(frame.groupby("permutations").agg(rmse=("rmse", "median"), top1=("top1", "mean"), coalition_fraction=("coalition_fraction", "mean")).reset_index().to_dict("records"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
