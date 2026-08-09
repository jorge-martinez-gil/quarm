"""Ablate repeat pairing without rerunning the expensive SQL workloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quarm.attribution import shapley_values
from quarm.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "icde_results"
PLAYERS = (
    "missing_measures",
    "value_noise",
    "orphan_foreign_keys",
    "stale_timestamps",
    "dimension_drift",
    "duplicate_facts",
)


def bootstrap_widths(frame: pd.DataFrame, *, draws: int, seed: int) -> list[dict[str, float | str]]:
    pivot = frame.pivot(index="repeat", columns="coalition", values="loss")
    matrix = pivot.to_numpy()
    coalitions = []
    for key in pivot.columns:
        coalitions.append(frozenset() if key == "EMPTY" else frozenset(key.split("|")))
    repeats, coalition_count = matrix.shape
    rng = np.random.default_rng(seed)
    paired = np.empty((draws, len(PLAYERS)))
    unpaired = np.empty_like(paired)
    for draw in range(draws):
        shared = rng.integers(0, repeats, size=repeats)
        paired_values = dict(zip(coalitions, matrix[shared].mean(axis=0), strict=True))
        independent = np.empty(coalition_count)
        for column in range(coalition_count):
            rows = rng.integers(0, repeats, size=repeats)
            independent[column] = matrix[rows, column].mean()
        unpaired_values = dict(zip(coalitions, independent, strict=True))
        p = shapley_values(paired_values, PLAYERS)
        u = shapley_values(unpaired_values, PLAYERS)
        paired[draw] = [p[player] for player in PLAYERS]
        unpaired[draw] = [u[player] for player in PLAYERS]
    records = []
    for index, player in enumerate(PLAYERS):
        paired_width = float(np.quantile(paired[:, index], 0.975) - np.quantile(paired[:, index], 0.025))
        unpaired_width = float(np.quantile(unpaired[:, index], 0.975) - np.quantile(unpaired[:, index], 0.025))
        records.append(
            {
                "channel": player,
                "paired_width": paired_width,
                "unpaired_width": unpaired_width,
                "paired_unpaired_ratio": paired_width / unpaired_width,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20270807)
    args = parser.parse_args()
    if args.draws < 200:
        raise ValueError("use at least 200 bootstrap draws")
    source = pd.concat(
        [pd.read_csv(RESULTS / "relational_observations.csv"), pd.read_csv(RESULTS / "nyc_observations.csv")],
        ignore_index=True,
    )
    rows = []
    for index, (regime, frame) in enumerate(source.groupby("regime", sort=True)):
        for record in bootstrap_widths(frame, draws=args.draws, seed=args.seed + index):
            rows.append({"regime": regime, **record})
    result = pd.DataFrame(rows)
    out = RESULTS / "paired_design_ablation.csv"
    result.to_csv(out, index=False)
    summary = result.groupby("regime").paired_unpaired_ratio.agg(["median", "min", "max"]).reset_index()
    manifest = {
        "study": "QuaRM paired-versus-independent bootstrap design ablation",
        "configuration": vars(args),
        "interpretation": "ratio below one means repeat-level pairing narrows the 95% Shapley interval under the resampled design",
        "source": ["relational_observations.csv", "nyc_observations.csv"],
        "files": {str(out.relative_to(ROOT)).replace("\\", "/"): sha256_file(out)},
    }
    write_json(RESULTS / "paired_design_ablation_manifest.json", manifest)
    print(json.dumps(summary.to_dict("records"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
