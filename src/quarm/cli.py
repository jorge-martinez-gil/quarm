"""Command-line interface for CSV audits and the bundled study."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from .attribution import attribute_quality_debt, coalition_key, concentration_radius
from .corruptions import default_corruptions
from .evaluation import evaluate_risk_surface
from .io import observations_to_records, write_json


def _factory(task: str, model: str):
    if task == "classification" and model == "logistic":
        return lambda seed: LogisticRegression(max_iter=3000, random_state=seed)
    if task == "classification" and model == "forest":
        return lambda seed: RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=1
        )
    if task == "regression" and model == "ridge":
        return lambda seed: Ridge(alpha=1.0)
    if task == "regression" and model == "forest":
        return lambda seed: RandomForestRegressor(
            n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=1
        )
    raise ValueError(f"model {model!r} is incompatible with task {task!r}")


def audit(args: argparse.Namespace) -> int:
    data = pd.read_csv(args.csv)
    if args.target not in data:
        raise ValueError(f"target column {args.target!r} not found")
    y = data.pop(args.target).to_numpy()
    non_numeric = list(data.select_dtypes(exclude="number").columns)
    if non_numeric:
        raise ValueError(f"all features must be numeric; encode these columns first: {non_numeric}")
    channels = default_corruptions()
    study = evaluate_risk_surface(
        data.to_numpy(dtype=float),
        y,
        channels,
        _factory(args.task, args.model),
        task=args.task,
        severity=args.severity,
        repeats=args.repeats,
        test_size=args.test_size,
        master_seed=args.seed,
    )
    attribution = attribute_quality_debt(
        study.observations,
        study.channels,
        bootstrap_samples=args.bootstrap,
        seed=args.seed + 1,
    )
    payload = {
        "interpretation": (
            "Counterfactual susceptibility on a held-out split. Treat as observed quality debt only when "
            "the uncorrupted input and held-out distribution are trusted references."
        ),
        "configuration": {
            "source": str(Path(args.csv).resolve()),
            "target": args.target,
            "task": args.task,
            "model": args.model,
            "severity": args.severity,
            "repeats": args.repeats,
            "test_size": args.test_size,
            "seed": args.seed,
            "channels": list(study.channels),
        },
        "total_debt": attribution.total_debt,
        "efficiency_residual": attribution.efficiency_residual,
        "banzhaf_residual": attribution.banzhaf_residual,
        "current_state_gains": attribution.gains,
        "singleton_effects": attribution.singletons,
        "simultaneous_hoeffding_radius": concentration_radius(len(study.channels), args.repeats),
        "attributions": attribution.estimates.to_dict(orient="records"),
        "interactions": attribution.interactions.to_dict(orient="records"),
        "coalition_means": {
            coalition_key(key): value for key, value in attribution.coalition_means.items()
        },
        "observations": observations_to_records(study.observations),
    }
    write_json(args.output, payload)
    print(f"Wrote {Path(args.output).resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quarm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="estimate a risk surface for a numeric CSV")
    audit_parser.add_argument("csv")
    audit_parser.add_argument("--target", required=True)
    audit_parser.add_argument("--task", choices=["classification", "regression"], required=True)
    audit_parser.add_argument("--model", choices=["logistic", "ridge", "forest"], required=True)
    audit_parser.add_argument("--severity", type=float, default=0.15)
    audit_parser.add_argument("--repeats", type=int, default=12)
    audit_parser.add_argument("--test-size", type=float, default=0.30)
    audit_parser.add_argument("--bootstrap", type=int, default=2000)
    audit_parser.add_argument("--seed", type=int, default=20260807)
    audit_parser.add_argument("--output", default="quarm-audit.json")
    audit_parser.set_defaults(func=audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
