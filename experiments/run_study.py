"""Reproduce the QuaRM empirical study and its machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from quarm.attribution import attribute_quality_debt, coalition_key, concentration_radius
from quarm.benchmark import datasets, model_factories
from quarm.corruptions import default_corruptions
from quarm.evaluation import evaluate_risk_surface
from quarm.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
FIGURES = ROOT / "artifacts" / "figures"

COLORS = {
    "missing_cells": "#0072B2",
    "feature_noise": "#D55E00",
    "target_noise": "#CC79A7",
    "duplicate_rows": "#009E73",
}


def paired_width_ratio(observations: pd.DataFrame, players: tuple[str, ...], seed: int, draws: int = 600):
    """Ratio of paired to independently resampled Shapley interval widths."""
    repeats = sorted(observations["repeat"].unique())
    pivot = observations.pivot(index="repeat", columns="coalition", values="loss").reindex(repeats)
    matrix, columns = pivot.to_numpy(), list(pivot.columns)
    rng = np.random.default_rng(seed)
    paired = np.empty((draws, len(players)))
    unpaired = np.empty((draws, len(players)))
    from quarm.attribution import shapley_values

    for b in range(draws):
        shared_rows = rng.integers(0, len(repeats), len(repeats))
        paired_values = dict(zip(columns, matrix[shared_rows].mean(axis=0), strict=True))
        paired[b] = list(shapley_values(paired_values, players).values())
        independent_values = {}
        for j, coalition in enumerate(columns):
            rows = rng.integers(0, len(repeats), len(repeats))
            independent_values[coalition] = matrix[rows, j].mean()
        unpaired[b] = list(shapley_values(independent_values, players).values())
    p_width = np.quantile(paired, 0.975, axis=0) - np.quantile(paired, 0.025, axis=0)
    u_width = np.quantile(unpaired, 0.975, axis=0) - np.quantile(unpaired, 0.025, axis=0)
    return {player: float(p_width[i] / u_width[i]) for i, player in enumerate(players)}


def remediation_metrics(means, players, debts):
    full = frozenset(players)
    marginal = {p: means[full] - means[full - {p}] for p in players}
    singleton = {p: means[frozenset({p})] - means[frozenset()] for p in players}
    shapley_choice = max(players, key=lambda p: debts[p])
    singleton_choice = max(players, key=lambda p: singleton[p])
    oracle_choice = max(players, key=lambda p: marginal[p])
    return {
        "shapley_choice": shapley_choice,
        "shapley_realized_repair_gain": marginal[shapley_choice],
        "singleton_choice": singleton_choice,
        "singleton_realized_repair_gain": marginal[singleton_choice],
        "oracle_choice": oracle_choice,
        "oracle_repair_gain": marginal[oracle_choice],
        "uniform_expected_repair_gain": float(np.mean(list(marginal.values()))),
        "full_context_marginals": marginal,
        "singletons": singleton,
    }


def plot_attributions(attributions: pd.DataFrame) -> Path:
    settings = list(dict.fromkeys(attributions["setting"]))
    channels = list(COLORS)
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.3), sharey=True)
    for ax, setting in zip(axes.flat, settings, strict=True):
        part = attributions[attributions["setting"] == setting].set_index("channel").reindex(channels)
        x = np.arange(len(channels))
        y = part["debt"].to_numpy() * 100
        low = (part["debt"] - part["ci_low"]).to_numpy() * 100
        high = (part["ci_high"] - part["debt"]).to_numpy() * 100
        ax.bar(x, y, color=[COLORS[c] for c in channels], alpha=0.9)
        ax.errorbar(x, y, yerr=np.vstack([low, high]), fmt="none", ecolor="#222222", capsize=2, lw=1)
        ax.axhline(0, color="#333333", lw=0.7)
        ax.set_title(setting.replace("__", " / "), fontsize=9, fontweight="bold")
        ax.set_xticks(x, ["missing", "feature\nnoise", "target\nnoise", "duplicates"], fontsize=7)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Attributed quality debt\n(percentage-point bounded risk)")
    axes[1, 0].set_ylabel("Attributed quality debt\n(percentage-point bounded risk)")
    fig.suptitle("QuaRM separates equal-prevalence defects by downstream harm", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012, "Bars: exact Shapley allocation; whiskers: paired 95% bootstrap intervals", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = FIGURES / "figure2_attributions.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_interactions(interactions: pd.DataFrame) -> Path:
    pivot = interactions.pivot_table(index="setting", columns="pair", values="interaction").fillna(0) * 100
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-max(1, abs(pivot.to_numpy()).max()), vmax=max(1, abs(pivot.to_numpy()).max()))
    ax.set_yticks(range(len(pivot)), [s.replace("__", " / ") for s in pivot.index], fontsize=8)
    ax.set_xticks(range(len(pivot.columns)), [p.replace("+", "\n+") for p in pivot.columns], fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Pairwise Shapley interactions reveal non-additive defect harm", fontweight="bold")
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("Interaction (risk percentage points)")
    fig.tight_layout()
    path = FIGURES / "figure3_interactions.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dose_response(dose: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for channel, part in dose.groupby("channel"):
        part = part.sort_values("severity")
        ax.plot(part["severity"] * 100, part["debt"] * 100, marker="o", lw=2, label=channel.replace("_", " "), color=COLORS[channel])
        ax.fill_between(part["severity"] * 100, part["ci_low"] * 100, part["ci_high"] * 100, color=COLORS[channel], alpha=0.14)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Injected prevalence (%)")
    ax.set_ylabel("Attributed quality debt (risk percentage points)")
    ax.set_title("Quality debt is a response curve, not a defect count", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "figure4_dose_response.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--dose-repeats", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--severity", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    observation_frames, attribution_frames, interaction_frames, summaries = [], [], [], []
    for dataset in datasets().values():
        for model_name, factory in model_factories(dataset.task).items():
            setting = f"{dataset.name}__{model_name}"
            print(f"[QuaRM] {setting}", flush=True)
            study = evaluate_risk_surface(
                dataset.X,
                dataset.y,
                default_corruptions(),
                factory,
                task=dataset.task,
                severity=args.severity,
                repeats=args.repeats,
                master_seed=args.seed,
            )
            result = attribute_quality_debt(
                study.observations,
                study.channels,
                bootstrap_samples=args.bootstrap,
                seed=args.seed + 17,
            )
            obs = study.observations.drop(columns=["diagnostics"]).copy()
            obs["coalition"] = obs["coalition"].map(coalition_key)
            obs.insert(0, "setting", setting)
            obs.insert(1, "dataset", dataset.name)
            obs.insert(2, "model", model_name)
            observation_frames.append(obs)

            attrs = result.estimates.copy()
            attrs.insert(0, "setting", setting)
            attrs.insert(1, "dataset", dataset.name)
            attrs.insert(2, "model", model_name)
            attribution_frames.append(attrs)
            ints = result.interactions.copy()
            ints["pair"] = ints["left"] + "+" + ints["right"]
            ints.insert(0, "setting", setting)
            interaction_frames.append(ints)

            debts = dict(zip(result.estimates["channel"], result.estimates["debt"], strict=True))
            remediation = remediation_metrics(result.coalition_means, study.channels, debts)
            clean = result.coalition_means[frozenset()]
            full = result.coalition_means[frozenset(study.channels)]
            width_ratios = paired_width_ratio(study.observations, study.channels, args.seed + 31)
            summaries.append(
                {
                    "setting": setting,
                    "dataset": dataset.name,
                    "model": model_name,
                    "task": dataset.task,
                    "n_rows": len(dataset.y),
                    "n_features": dataset.X.shape[1],
                    "clean_loss": clean,
                    "fully_corrupted_loss": full,
                    "total_debt": result.total_debt,
                    "efficiency_residual": result.efficiency_residual,
                    "top_channel": result.estimates.iloc[0]["channel"],
                    "top_probability": result.estimates.iloc[0]["rank_1_probability"],
                    "max_abs_interaction": float(result.interactions["interaction"].abs().max()),
                    "median_paired_unpaired_width_ratio": float(np.median(list(width_ratios.values()))),
                    "hoeffding_radius": concentration_radius(len(study.channels), args.repeats),
                    **{k: v for k, v in remediation.items() if not isinstance(v, dict)},
                    "provenance": dataset.provenance,
                }
            )

    observations = pd.concat(observation_frames, ignore_index=True)
    attributions = pd.concat(attribution_frames, ignore_index=True)
    interactions = pd.concat(interaction_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)

    # Dose response on the largest controlled classification setting.
    dose_frames = []
    dataset = datasets()["synthetic_nonlinear"]
    factory = model_factories(dataset.task)["logistic"]
    for severity in [0.05, 0.10, 0.20, 0.30]:
        print(f"[QuaRM] dose response {severity:.2f}", flush=True)
        study = evaluate_risk_surface(
            dataset.X,
            dataset.y,
            default_corruptions(),
            factory,
            task=dataset.task,
            severity=severity,
            repeats=args.dose_repeats,
            master_seed=args.seed,
        )
        result = attribute_quality_debt(
            study.observations,
            study.channels,
            bootstrap_samples=args.bootstrap,
            seed=args.seed + 53,
        )
        frame = result.estimates.copy()
        frame.insert(0, "severity", severity)
        dose_frames.append(frame)
    dose = pd.concat(dose_frames, ignore_index=True)

    paths = {
        "observations": RESULTS / "coalition_observations.csv",
        "attributions": RESULTS / "attributions.csv",
        "interactions": RESULTS / "interactions.csv",
        "summary": RESULTS / "study_summary.csv",
        "dose": RESULTS / "dose_response.csv",
    }
    observations.to_csv(paths["observations"], index=False)
    attributions.to_csv(paths["attributions"], index=False)
    interactions.to_csv(paths["interactions"], index=False)
    summary.to_csv(paths["summary"], index=False)
    dose.to_csv(paths["dose"], index=False)
    figure_paths = [plot_attributions(attributions), plot_interactions(interactions), plot_dose_response(dose)]

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        revision = "unversioned-workspace"
    manifest_path = RESULTS / "manifest.json"
    manifest = {
        "study": "QuaRM full-factorial quality-debt benchmark",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "configuration": vars(args),
        "environment": environment,
        "git_revision": revision,
        "runtime_seconds": time.perf_counter() - start,
        "design": {
            "loss": "classification error or clipped IQR-normalized absolute regression error, both in [0,1]",
            "test_distribution": "clean held-out 30% split",
            "pairing": "same split and model seed across every coalition within a repeat",
            "corruption_streams": "SHA-256-derived channel-specific seeds invariant to coalition membership",
            "coalitions_per_setting": 16,
        },
        "files": {},
    }
    for path in [*paths.values(), *figure_paths]:
        manifest["files"][str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    write_json(manifest_path, manifest)
    print(f"[QuaRM] completed in {manifest['runtime_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
