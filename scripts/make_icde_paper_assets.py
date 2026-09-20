"""Generate evidence-linked figures and LaTeX macros for the ICDE manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "icde2027"
FIGURES = ROOT / "artifacts" / "icde_figures"
RESULTS = ROOT / "artifacts" / "icde_results"


def _box(ax, xy, width, height, text, *, face, edge="#263238", size=8.2, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.016,rounding_size=0.025",
        linewidth=1.1,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=size, weight=weight)
    return patch


def _arrow(ax, start, end, *, color="#455A64", style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, lw=1.15, color=color))


def concept_figure() -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.15, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.01, 0.61), 0.18, 0.25, "Reference snapshot\n$D^*$", face="#E3F2FD", weight="bold")
    _box(ax, (0.01, 0.28), 0.18, 0.25, r"Analytical workload" + "\n" + r"$W,\,\ell$", face="#E8F5E9", weight="bold")
    _box(ax, (0.01, 0.02), 0.18, 0.17, r"Mechanisms" + "\n" + r"$\mathcal{M}$", face="#FFF3E0", weight="bold")

    _box(ax, (0.29, 0.31), 0.25, 0.37, "Paired coalition executor\n\nExact: $2^m$ coalitions\nSampled: permutation prefixes", face="#F3E5F5", weight="bold")
    for y in (0.735, 0.405, 0.105):
        _arrow(ax, (0.19, y), (0.29, 0.50))

    _box(ax, (0.62, 0.36), 0.15, 0.28, r"Risk surface" + "\n\n" + r"$R(S)$ for" + "\n" + r"$S\subseteq\mathcal{M}$", face="#E0F2F1", weight="bold")
    _arrow(ax, (0.54, 0.50), (0.62, 0.50))

    outputs = [
        (0.81, 0.70, "Accounting", r"$\phi_i$"),
        (0.81, 0.39, "Interactions", r"$I_{ij}$"),
        (0.81, 0.08, "Intervention", r"$g_i$"),
    ]
    for x, y, label, symbol in outputs:
        _box(ax, (x, y), 0.18, 0.20, f"{label}\n{symbol}", face="#FAFAFA", weight="bold")
        _arrow(ax, (0.77, 0.50), (x, y + 0.10))
    ax.text(0.695, 0.24, "one surface,\nthree questions", ha="center", va="center", fontsize=7.5, color="#37474F")
    fig.tight_layout(pad=0.25)
    pdf = FIGURES / "icde_figure1_concept.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def decision_gap_figure() -> list[Path]:
    channels = [
        "missing_measures",
        "value_noise",
        "orphan_foreign_keys",
        "stale_timestamps",
        "dimension_drift",
        "duplicate_facts",
    ]
    labels = ["missing", "value\nnoise", "orphan\nkeys", "stale\ntime", "dimension\ndrift", "duplicate\nfacts"]
    attributes = pd.concat(
        [pd.read_csv(RESULTS / "relational_attributions.csv"), pd.read_csv(RESULTS / "nyc_attributions.csv")],
        ignore_index=True,
    )
    observations = pd.concat(
        [pd.read_csv(RESULTS / "relational_observations.csv"), pd.read_csv(RESULTS / "nyc_observations.csv")],
        ignore_index=True,
    )
    regimes = ["uniform", "skewed", "seasonal", "nyc_taxi"]
    titles = ["Uniform", "Skewed", "Seasonal", "NYC Taxi"]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.75), sharey=True)
    for ax, regime, title in zip(axes.flat, regimes, titles, strict=True):
        means = observations[observations.regime == regime].groupby("coalition").loss.mean().to_dict()
        full_key = "|".join(sorted(channels))
        marginal = []
        for item in channels:
            without = "|".join(sorted(set(channels) - {item}))
            marginal.append(means[full_key] - means[without])
        shapley = attributes[attributes.regime == regime].set_index("channel").reindex(channels).debt.to_numpy()
        x = range(len(channels))
        ax.bar([value - 0.19 for value in x], 100 * shapley, width=0.38, color="#0072B2", label="Shapley account $\\phi_i$")
        ax.bar([value + 0.19 for value in x], 100 * pd.Series(marginal), width=0.38, color="#D55E00", label="current repair gain $g_i$")
        ax.axhline(0, color="#263238", lw=0.75)
        ax.set_title(title, weight="bold", fontsize=9.5)
        ax.set_xticks(list(x), labels, fontsize=6.3)
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].set_ylabel("Workload risk points")
    axes[1, 0].set_ylabel("Workload risk points")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.7)
    pdf = FIGURES / "icde_figure4_accounting_decision.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def fmt(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def generated_tex() -> Path:
    relational = pd.read_csv(RESULTS / "relational_summary.csv")
    nyc = pd.read_csv(RESULTS / "nyc_summary.csv")
    combined = pd.concat([nyc, relational], ignore_index=True)
    attrs = pd.concat(
        [pd.read_csv(RESULTS / "nyc_attributions.csv"), pd.read_csv(RESULTS / "relational_attributions.csv")],
        ignore_index=True,
    )
    interactions = pd.concat(
        [pd.read_csv(RESULTS / "nyc_interactions.csv"), pd.read_csv(RESULTS / "relational_interactions.csv")],
        ignore_index=True,
    )
    sampling = pd.read_csv(RESULTS / "sampling_scalability_m12.csv")
    ml = pd.read_csv(ROOT / "artifacts" / "results" / "study_summary.csv")

    names = {"nyc_taxi": "NYC Taxi", "uniform": "Uniform", "skewed": "Skewed", "seasonal": "Seasonal"}
    channel = {
        "missing_measures": "missing measures",
        "value_noise": "value noise",
        "orphan_foreign_keys": "orphan keys",
        "stale_timestamps": "stale time",
        "dimension_drift": "dimension drift",
        "duplicate_facts": "duplicate facts",
    }
    lines = ["% Generated by scripts/make_icde_paper_assets.py; do not edit manually."]
    nyc_attr = attrs[(attrs.regime == "nyc_taxi") & (attrs.channel == "dimension_drift")].iloc[0]
    lines.extend(
        [
            rf"\newcommand{{\NYCRisk}}{{{fmt(float(nyc.iloc[0].full_risk))}}}",
            rf"\newcommand{{\NYCDimensionDebt}}{{{fmt(float(nyc_attr.debt))}}}",
            rf"\newcommand{{\NYCDimensionLow}}{{{fmt(float(nyc_attr.ci_low))}}}",
            rf"\newcommand{{\NYCDimensionHigh}}{{{fmt(float(nyc_attr.ci_high))}}}",
            rf"\newcommand{{\MaxRelInteraction}}{{{fmt(float(interactions.interaction.abs().max()))}}}",
            rf"\newcommand{{\MLPositive}}{{{int((ml.total_debt > 0).sum())}}}",
            rf"\newcommand{{\MLSettings}}{{{len(ml)}}}",
            rf"\newcommand{{\MLMaxInteraction}}{{{fmt(float(ml.max_abs_interaction.max()))}}}",
        ]
    )
    word = {64: "SixtyFour", 128: "OneTwentyEight"}
    for permutations in (64, 128):
        part = sampling[sampling.permutations == permutations]
        lines.extend(
            [
                rf"\newcommand{{\SamplingTop{word[permutations]}}}{{{100 * part.top1.mean():.1f}}}",
                rf"\newcommand{{\SamplingSurface{word[permutations]}}}{{{100 * part.coalition_fraction.mean():.1f}}}",
                rf"\newcommand{{\SamplingRMSE{word[permutations]}}}{{{100 * part.rmse.median():.3f}}}",
            ]
        )

    table = [
        r"% Generated by scripts/make_icde_paper_assets.py; do not edit manually.",
        r"\begin{table*}[t]",
        r"\caption{Relational results at 10\% prevalence. Risk, allocations, and gains are percentage points of bounded workload-answer loss. $\phi$ is the top Shapley account; $g^*$ is the best current-state repair. Negative $g(\arg\max\phi)$ means that repairing the largest aggregate account alone worsens the current joint state.}",
        r"\label{tab:relational}",
        r"\centering\small\setlength{\tabcolsep}{4.3pt}",
        r"\begin{tabular}{lrrrrlllrr}",
        r"\toprule",
        r"Reference & rows & $R(\mathcal{M})$ & $Q(\mathcal{M})$ & $\max|I_{ij}|$ & top $\phi$ & Shapley repair & best repair & $g(\arg\max\phi)$ & $g^*$ \\",
        r"\midrule",
    ]
    for row in combined.itertuples():
        table.append(
            f"{names[row.regime]} & {int(row.orders):,} & {fmt(row.full_risk)} & {fmt(row.total_debt)} & "
            f"{fmt(row.max_abs_interaction)} & {channel[row.top_channel]} & {channel[row.shapley_choice]} & "
            f"{channel[row.oracle_choice]} & {fmt(row.shapley_gain)} & {fmt(row.oracle_gain)} \\\\".replace(",", "{,}")
        )
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

    out = PAPER / "generated_results.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PAPER / "generated_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    allocations = ["% Generated by scripts/make_icde_paper_assets.py; do not edit manually."]
    allocations.extend(
        [
            r"\begin{table*}[t]",
            r"\caption{Complete relational Shapley allocations. Values and intervals are workload risk points; rank is the paired-bootstrap probability of being the largest account.}",
            r"\label{tab:allocations}",
            r"\centering\small\setlength{\tabcolsep}{6pt}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Reference & mechanism & debt & 95\% low & 95\% high & rank-one (\%) \\",
            r"\midrule",
        ]
    )
    for regime in ["nyc_taxi", "uniform", "skewed", "seasonal"]:
        part = attrs[attrs.regime == regime].sort_values("debt", ascending=False)
        for row in part.itertuples():
            allocations.append(
                f"{names[regime]} & {channel[row.channel]} & {fmt(row.debt)} & {fmt(row.ci_low)} & "
                f"{fmt(row.ci_high)} & {100 * row.rank_1_probability:.1f} \\\\"
            )
        allocations.append(r"\addlinespace")
    allocations.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

    ml_table = [
        r"% Generated by scripts/make_icde_paper_assets.py; do not edit manually.",
        r"\begin{table*}[t]",
        r"\caption{Complete ML confirmation study at 15\% prevalence. Debt and maximum interaction are bounded clean-test risk points.}",
        r"\label{tab:ml}",
        r"\centering\small\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrlrrl}",
        r"\toprule",
        r"Dataset & model & rows & debt & top account & top prob. (\%) & $\max|I|$ & current-state best \\",
        r"\midrule",
    ]
    for row in ml.itertuples():
        ml_table.append(
            f"{str(row.dataset).replace('_', ' ')} & {row.model} & {int(row.n_rows)} & {fmt(row.total_debt)} & "
            f"{str(row.top_channel).replace('_', ' ')} & {100 * row.top_probability:.1f} & "
            f"{fmt(row.max_abs_interaction)} & {str(row.oracle_choice).replace('_', ' ')} \\\\"
        )
    ml_table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

    sampling_table = [
        r"% Generated by scripts/make_icde_paper_assets.py; do not edit manually.",
        r"\begin{table}[t]",
        r"\caption{Controlled 12-mechanism permutation study (400 trials per budget).}",
        r"\label{tab:sampling}",
        r"\centering\small\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$B$ & median RMSE & top-one (\%) & surface (\%) \\",
        r"\midrule",
    ]
    for permutations, part in sampling.groupby("permutations", sort=True):
        sampling_table.append(
            f"{int(permutations)} & {100 * part.rmse.median():.3f} & {100 * part.top1.mean():.1f} & "
            f"{100 * part.coalition_fraction.mean():.1f} \\\\"
        )
    sampling_table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (PAPER / "generated_allocations_table.tex").write_text("\n".join(allocations) + "\n", encoding="utf-8")
    (PAPER / "generated_ml_table.tex").write_text("\n".join(ml_table) + "\n", encoding="utf-8")
    (PAPER / "generated_supplement_tables.tex").write_text(
        "% Deprecated: the evidence tables moved into the main paper as\n"
        "% generated_allocations_table.tex and generated_ml_table.tex.\n",
        encoding="utf-8",
    )
    (PAPER / "generated_sampling_table.tex").write_text("\n".join(sampling_table) + "\n", encoding="utf-8")
    return out


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    concept_figure()
    decision_gap_figure()
    generated_tex()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
