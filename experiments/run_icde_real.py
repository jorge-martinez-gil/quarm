"""Run QuaRM on the official NYC TLC January 2025 relational snapshot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quarm.attribution import attribute_quality_debt, coalition_key
from quarm.io import sha256_file, write_json
from quarm.relational import default_relational_corruptions, evaluate_relational_surface, load_nyc_taxi_star_schema
from quarm.sampling import banzhaf_values

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "icde_results"
FIGURES = ROOT / "artifacts" / "icde_figures"
RAW = ROOT / "data" / "raw"

COLORS = {
    "missing_measures": "#0072B2",
    "value_noise": "#D55E00",
    "orphan_foreign_keys": "#CC79A7",
    "stale_timestamps": "#E69F00",
    "dimension_drift": "#56B4E9",
    "duplicate_facts": "#009E73",
}


def decision_metrics(means, players, debts):
    full = frozenset(players)
    marginal = {player: means[full] - means[full - {player}] for player in players}
    singleton = {player: means[frozenset({player})] - means[frozenset()] for player in players}
    choices = {
        "shapley_choice": max(players, key=debts.get),
        "singleton_choice": max(players, key=singleton.get),
        "oracle_choice": max(players, key=marginal.get),
    }
    return {
        **choices,
        "shapley_gain": marginal[choices["shapley_choice"]],
        "singleton_gain": marginal[choices["singleton_choice"]],
        "uniform_gain": float(np.mean(list(marginal.values()))),
        "oracle_gain": marginal[choices["oracle_choice"]],
    }


def combined_plot(nyc: pd.DataFrame) -> list[Path]:
    synthetic = pd.read_csv(RESULTS / "relational_attributions.csv")
    combined = pd.concat([synthetic, nyc], ignore_index=True)
    regimes = ["uniform", "skewed", "seasonal", "nyc_taxi"]
    channels = list(COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 6.5), sharey=True)
    for ax, regime in zip(axes.flat, regimes, strict=True):
        part = combined[combined["regime"] == regime].set_index("channel").reindex(channels)
        x = np.arange(len(channels))
        y = 100 * part["debt"].to_numpy()
        error = np.vstack([100 * (part["debt"] - part["ci_low"]), 100 * (part["ci_high"] - part["debt"])])
        ax.bar(x, y, color=[COLORS[channel] for channel in channels], width=0.74)
        ax.errorbar(x, y, yerr=error, fmt="none", ecolor="#222222", capsize=2, lw=0.8)
        ax.axhline(0, color="#333333", lw=0.7)
        ax.set_title("NYC Taxi" if regime == "nyc_taxi" else regime.capitalize(), fontweight="bold")
        ax.set_xticks(x, ["missing", "value\nnoise", "orphan\nFK", "stale\ntime", "dimension\ndrift", "duplicate\nfacts"], fontsize=7)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Attributed answer debt (risk points)")
    axes[1, 0].set_ylabel("Attributed answer debt (risk points)")
    fig.suptitle("Equal prevalence is not equal harm across relational populations", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf = FIGURES / "icde_figure2_all_relational_attributions.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--severity", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20270807)
    args = parser.parse_args()
    parquet = RAW / "yellow_tripdata_2025-01.parquet"
    zones = RAW / "taxi_zone_lookup.csv"
    if not parquet.exists() or not zones.exists():
        raise FileNotFoundError("run scripts/fetch_public_data.py first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    schema = load_nyc_taxi_star_schema(str(parquet), str(zones), n_orders=args.orders)
    channels = default_relational_corruptions()
    started = time.perf_counter()
    print(f"[ICDE] NYC TLC / {args.orders:,} trips", flush=True)
    study = evaluate_relational_surface(
        schema, channels, severity=args.severity, repeats=args.repeats, master_seed=args.seed
    )
    elapsed = time.perf_counter() - started
    result = attribute_quality_debt(
        study.observations,
        study.channels,
        bootstrap_samples=args.bootstrap,
        seed=args.seed + 91,
    )
    attrs = result.estimates.copy()
    attrs.insert(0, "regime", "nyc_taxi")
    interactions = result.interactions.copy()
    interactions.insert(0, "regime", "nyc_taxi")
    observations = study.observations.drop(columns=["diagnostics", "query_losses"]).copy()
    observations.insert(0, "regime", "nyc_taxi")
    observations["coalition"] = observations["coalition"].map(coalition_key)
    query_rows = []
    for row in study.observations.itertuples():
        for query, loss in row.query_losses.items():
            query_rows.append({"repeat": row.repeat, "coalition": row.coalition, "query": query, "loss": loss})
    query_frame = pd.DataFrame(query_rows)
    query_attrs = []
    for query, frame in query_frame.groupby("query"):
        allocation = attribute_quality_debt(
            frame[["repeat", "coalition", "loss"]],
            study.channels,
            bootstrap_samples=max(500, args.bootstrap // 2),
            seed=args.seed + 92,
        ).estimates
        allocation.insert(0, "regime", "nyc_taxi")
        allocation.insert(1, "query", query)
        query_attrs.append(allocation)
    query_attrs = pd.concat(query_attrs, ignore_index=True)
    debts = dict(zip(attrs["channel"], attrs["debt"], strict=True))
    banzhaf = banzhaf_values(result.coalition_means, study.channels)
    summary = {
        "regime": "nyc_taxi",
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
        "max_abs_interaction": float(interactions["interaction"].abs().max()),
        "top_channel": attrs.iloc[0]["channel"],
        "top_probability": attrs.iloc[0]["rank_1_probability"],
        **decision_metrics(result.coalition_means, study.channels, debts),
    }
    paths = {
        "observations": RESULTS / "nyc_observations.csv",
        "attributions": RESULTS / "nyc_attributions.csv",
        "interactions": RESULTS / "nyc_interactions.csv",
        "query_attributions": RESULTS / "nyc_query_attributions.csv",
        "summary": RESULTS / "nyc_summary.csv",
    }
    observations.to_csv(paths["observations"], index=False)
    attrs.to_csv(paths["attributions"], index=False)
    interactions.to_csv(paths["interactions"], index=False)
    query_attrs.to_csv(paths["query_attributions"], index=False)
    pd.DataFrame([summary]).to_csv(paths["summary"], index=False)
    figures = combined_plot(attrs)
    manifest = {
        "dataset": {
            "name": "NYC TLC Yellow Taxi Trip Records, January 2025",
            "source": "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
            "parquet_sha256": sha256_file(parquet),
            "zone_lookup_sha256": sha256_file(zones),
            "reference_status": "real observed snapshot used as a controlled susceptibility reference; not asserted clean",
        },
        "configuration": vars(args),
        "runtime_seconds": elapsed,
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in [*paths.values(), *figures]
        },
    }
    write_json(RESULTS / "nyc_manifest.json", manifest)
    print(f"[ICDE] NYC completed in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
