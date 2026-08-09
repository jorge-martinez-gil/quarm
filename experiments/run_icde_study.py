"""ICDE 2027 study: relational workloads, approximation, and system scale."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quarm.attribution import attribute_quality_debt, coalition_key, shapley_values
from quarm.io import sha256_file, write_json
from quarm.relational import (
    default_relational_corruptions,
    evaluate_relational_surface,
    generate_star_schema,
)
from quarm.sampling import banzhaf_values, estimate_permutation_shapley

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "icde_results"
FIGURES = ROOT / "artifacts" / "icde_figures"

COLORS = {
    "missing_measures": "#0072B2",
    "value_noise": "#D55E00",
    "orphan_foreign_keys": "#CC79A7",
    "stale_timestamps": "#E69F00",
    "dimension_drift": "#56B4E9",
    "duplicate_facts": "#009E73",
}


def remediation(means, players, shapley):
    full = frozenset(players)
    marginal = {player: means[full] - means[full - {player}] for player in players}
    singleton = {player: means[frozenset({player})] - means[frozenset()] for player in players}
    shapley_choice = max(players, key=shapley.get)
    singleton_choice = max(players, key=singleton.get)
    oracle_choice = max(players, key=marginal.get)
    return {
        "shapley_choice": shapley_choice,
        "singleton_choice": singleton_choice,
        "oracle_choice": oracle_choice,
        "shapley_gain": marginal[shapley_choice],
        "singleton_gain": marginal[singleton_choice],
        "uniform_gain": float(np.mean(list(marginal.values()))),
        "oracle_gain": marginal[oracle_choice],
    }


def plot_relational_attributions(frame: pd.DataFrame) -> Path:
    regimes = ["uniform", "skewed", "seasonal"]
    channels = list(COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), sharey=True)
    for ax, regime in zip(axes, regimes, strict=True):
        part = frame[frame["regime"] == regime].set_index("channel").reindex(channels)
        x = np.arange(len(channels))
        y = 100 * part["debt"].to_numpy()
        error = np.vstack([100 * (part["debt"] - part["ci_low"]), 100 * (part["ci_high"] - part["debt"])])
        ax.bar(x, y, color=[COLORS[channel] for channel in channels], width=0.72)
        ax.errorbar(x, y, yerr=error, fmt="none", ecolor="#20242A", capsize=2, lw=0.9)
        ax.axhline(0, color="#333333", lw=0.7)
        ax.set_title(regime.capitalize(), fontweight="bold")
        ax.set_xticks(x, ["missing", "value\nnoise", "orphan\nFK", "stale\ntime", "dimension\ndrift", "duplicate\nfacts"], fontsize=7)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Attributed SQL answer debt (percentage points)")
    fig.suptitle("Equal violation prevalence produces workload-specific quality debt", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIGURES / "icde_figure2_relational_attributions.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_query_profiles(frame: pd.DataFrame) -> Path:
    regimes = ["uniform", "skewed", "seasonal"]
    channels = list(COLORS)
    queries = list(dict.fromkeys(frame["query"]))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.1), sharey=True)
    max_abs = max(0.01, float(frame["debt"].abs().max() * 100))
    for ax, regime in zip(axes, regimes, strict=True):
        pivot = frame[frame["regime"] == regime].pivot(index="channel", columns="query", values="debt").reindex(index=channels, columns=queries) * 100
        image = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_title(regime.capitalize(), fontweight="bold")
        ax.set_xticks(range(len(queries)), [query.replace("_", "\n") for query in queries], rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(channels)), [channel.replace("_", " ") for channel in channels], fontsize=7)
    bar = fig.colorbar(image, ax=axes, fraction=0.024, pad=0.02)
    bar.set_label("Query-specific debt (percentage points)")
    fig.suptitle("A single data defect has different consequences across SQL queries", fontweight="bold", fontsize=13)
    fig.subplots_adjust(left=0.12, right=0.91, bottom=0.25, top=0.86, wspace=0.12)
    path = FIGURES / "icde_figure3_query_profiles.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sampling(frame: pd.DataFrame) -> Path:
    grouped = frame.groupby("permutations").agg(
        rmse_median=("rmse", "median"),
        rmse_low=("rmse", lambda values: np.quantile(values, 0.10)),
        rmse_high=("rmse", lambda values: np.quantile(values, 0.90)),
        top1=("top1", "mean"),
        coalition_fraction=("coalition_fraction", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    x = grouped["permutations"]
    axes[0].plot(x, grouped["rmse_median"] * 100, marker="o", color="#0072B2", label="median RMSE")
    axes[0].fill_between(x, grouped["rmse_low"] * 100, grouped["rmse_high"] * 100, color="#0072B2", alpha=0.16, label="10-90%")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Sampled repair orders")
    axes[0].set_ylabel("Attribution RMSE (risk points)")
    axes[0].legend(frameon=False, fontsize=8)
    ax2 = axes[1]
    ax2.plot(x, grouped["top1"] * 100, marker="o", color="#009E73", label="top-1 recovery")
    ax2.plot(x, grouped["coalition_fraction"] * 100, marker="s", color="#D55E00", label="coalitions evaluated")
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Sampled repair orders")
    ax2.set_ylabel("Percent")
    ax2.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Permutation QuaRM trades coalition cost for controlled approximation", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIGURES / "icde_figure4_sampling.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scale(frame: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(frame["orders"], frame["seconds"], marker="o", color="#7A5195", lw=2)
    for row in frame.itertuples():
        ax.annotate(f"{row.seconds:.1f}s", (row.orders, row.seconds), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fact rows")
    ax.set_ylabel("One-repeat exact 6-channel surface (s)")
    ax.set_title("End-to-end relational surface runtime", fontweight="bold")
    ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    path = FIGURES / "icde_figure5_scale.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--severity", type=float, default=0.10)
    parser.add_argument("--sampling-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20270807)
    parser.add_argument("--scale-max", type=int, default=1_000_000)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    channels = default_relational_corruptions()
    start_all = time.perf_counter()
    observation_frames, attribution_frames, interaction_frames = [], [], []
    query_frames, summaries, exact_games = [], [], {}

    for regime_index, regime in enumerate(["uniform", "skewed", "seasonal"]):
        print(f"[ICDE] relational {regime} / {args.orders:,} orders", flush=True)
        schema = generate_star_schema(args.orders, regime=regime, seed=args.seed + regime_index, name=f"retail_{regime}")
        started = time.perf_counter()
        study = evaluate_relational_surface(
            schema,
            channels,
            severity=args.severity,
            repeats=args.repeats,
            master_seed=args.seed,
        )
        elapsed = time.perf_counter() - started
        result = attribute_quality_debt(
            study.observations,
            study.channels,
            bootstrap_samples=args.bootstrap,
            seed=args.seed + 100 + regime_index,
        )
        exact_games[regime] = result.coalition_means
        obs = study.observations.copy()
        obs.insert(0, "regime", regime)
        for row in obs.itertuples():
            for query, loss in row.query_losses.items():
                query_frames.append(
                    {"regime": regime, "repeat": row.repeat, "coalition": row.coalition, "query": query, "loss": loss}
                )
        clean_obs = obs.drop(columns=["diagnostics", "query_losses"])
        clean_obs["coalition"] = clean_obs["coalition"].map(coalition_key)
        observation_frames.append(clean_obs)
        attrs = result.estimates.copy()
        attrs.insert(0, "regime", regime)
        attribution_frames.append(attrs)
        ints = result.interactions.copy()
        ints.insert(0, "regime", regime)
        interaction_frames.append(ints)
        shapley = dict(zip(result.estimates["channel"], result.estimates["debt"], strict=True))
        banzhaf = banzhaf_values(result.coalition_means, study.channels)
        decision = remediation(result.coalition_means, study.channels, shapley)
        summaries.append(
            {
                "regime": regime,
                "orders": args.orders,
                "customers": len(schema.customers),
                "products": len(schema.products),
                "queries": len(study.workload),
                "coalitions": 2 ** len(study.channels),
                "repeats": args.repeats,
                "wall_seconds": elapsed,
                "full_risk": result.coalition_means[frozenset(study.channels)],
                "total_debt": result.total_debt,
                "shapley_residual": result.efficiency_residual,
                "banzhaf_residual": sum(banzhaf.values()) - result.total_debt,
                "max_abs_interaction": float(result.interactions["interaction"].abs().max()),
                "top_channel": result.estimates.iloc[0]["channel"],
                "top_probability": result.estimates.iloc[0]["rank_1_probability"],
                **decision,
            }
        )

    observations = pd.concat(observation_frames, ignore_index=True)
    attributions = pd.concat(attribution_frames, ignore_index=True)
    interactions = pd.concat(interaction_frames, ignore_index=True)
    query_observations = pd.DataFrame(query_frames)
    summary = pd.DataFrame(summaries)

    print("[ICDE] query-specific allocations", flush=True)
    query_attributions = []
    for (regime, query), frame in query_observations.groupby(["regime", "query"]):
        result = attribute_quality_debt(
            frame[["repeat", "coalition", "loss"]],
            tuple(channel.name for channel in channels),
            bootstrap_samples=max(500, args.bootstrap // 2),
            seed=args.seed + 300,
        )
        part = result.estimates.copy()
        part.insert(0, "regime", regime)
        part.insert(1, "query", query)
        query_attributions.append(part)
    query_attributions = pd.concat(query_attributions, ignore_index=True)

    print("[ICDE] permutation approximation", flush=True)
    sampling_rows = []
    players = tuple(channel.name for channel in channels)
    for regime, means in exact_games.items():
        exact = shapley_values(means, players)
        exact_vector = np.array([exact[player] for player in players])
        exact_top = players[int(np.argmax(exact_vector))]
        for permutations in [2, 4, 8, 16, 32, 64, 128]:
            for trial in range(args.sampling_trials):
                estimate = estimate_permutation_shapley(
                    means.__getitem__, players, permutations=permutations, seed=args.seed + trial + 10_000 * permutations
                )
                vector = np.array([estimate.values[player] for player in players])
                sampling_rows.append(
                    {
                        "regime": regime,
                        "permutations": permutations,
                        "trial": trial,
                        "rmse": float(np.sqrt(np.mean((vector - exact_vector) ** 2))),
                        "max_error": float(np.max(np.abs(vector - exact_vector))),
                        "top1": players[int(np.argmax(vector))] == exact_top,
                        "unique_coalitions": estimate.unique_coalitions,
                        "coalition_fraction": estimate.unique_coalitions / estimate.exact_coalitions,
                        "hoeffding_radius": estimate.hoeffding_radius,
                    }
                )
    sampling = pd.DataFrame(sampling_rows)

    print("[ICDE] scale study", flush=True)
    scale_rows = []
    scale_sizes = [10_000, 100_000]
    if args.scale_max >= 1_000_000:
        scale_sizes.append(1_000_000)
    elif args.scale_max not in scale_sizes:
        scale_sizes.append(args.scale_max)
    for size in sorted(set(scale_sizes)):
        schema = generate_star_schema(size, regime="uniform", seed=args.seed)
        started = time.perf_counter()
        scale_study = evaluate_relational_surface(
            schema, channels, severity=args.severity, repeats=1, master_seed=args.seed
        )
        elapsed = time.perf_counter() - started
        scale_rows.append(
            {
                "orders": size,
                "seconds": elapsed,
                "coalition_evaluations": len(scale_study.observations),
                "query_executions": len(scale_study.observations) * len(scale_study.workload),
                "peak_risk": float(scale_study.observations["loss"].max()),
            }
        )
        print(f"[ICDE] scale {size:,}: {elapsed:.1f}s", flush=True)
    scale = pd.DataFrame(scale_rows)

    files = {
        "relational_observations": RESULTS / "relational_observations.csv",
        "relational_attributions": RESULTS / "relational_attributions.csv",
        "relational_interactions": RESULTS / "relational_interactions.csv",
        "query_attributions": RESULTS / "query_attributions.csv",
        "relational_summary": RESULTS / "relational_summary.csv",
        "sampling": RESULTS / "sampling_convergence.csv",
        "scale": RESULTS / "scale.csv",
    }
    observations.to_csv(files["relational_observations"], index=False)
    attributions.to_csv(files["relational_attributions"], index=False)
    interactions.to_csv(files["relational_interactions"], index=False)
    query_attributions.to_csv(files["query_attributions"], index=False)
    summary.to_csv(files["relational_summary"], index=False)
    sampling.to_csv(files["sampling"], index=False)
    scale.to_csv(files["scale"], index=False)
    figure_paths = [
        plot_relational_attributions(attributions),
        plot_query_profiles(query_attributions),
        plot_sampling(sampling),
        plot_scale(scale),
    ]
    generated = [*files.values()]
    for pdf_path in figure_paths:
        generated.extend([pdf_path, pdf_path.with_suffix(".png")])
    manifest = {
        "study": "QuaRM ICDE 2027 relational and approximation study",
        "configuration": vars(args),
        "runtime_seconds": time.perf_counter() - start_all,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "duckdb": duckdb.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "design": {
            "relations": ["customers", "products", "orders"],
            "workload_queries": 8,
            "channels": players,
            "loss": "mean bounded normalized-L1 query-answer error",
            "pairing": "channel-specific SHA-256 random streams shared across coalitions by repeat",
        },
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in generated
        },
    }
    manifest_path = RESULTS / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"[ICDE] completed in {manifest['runtime_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
