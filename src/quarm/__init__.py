"""QuaRM: counterfactual risk surfaces for data-quality measurement."""

from .attribution import (
    AttributionResult,
    attribute_quality_debt,
    coalition_key,
    concentration_radius,
    shapley_interactions,
    shapley_values,
)
from .corruptions import (
    Corruption,
    DuplicateRows,
    FeatureNoise,
    MissingCells,
    TargetNoise,
    default_corruptions,
)
from .evaluation import StudyResult, evaluate_risk_surface
from .relational import (
    RelationalStudyResult,
    StarSchema,
    default_relational_corruptions,
    evaluate_relational_surface,
    generate_star_schema,
    load_nyc_taxi_star_schema,
    normalize_query_weights,
)
from .sampling import PermutationEstimate, banzhaf_values, estimate_permutation_shapley

__all__ = [
    "AttributionResult",
    "Corruption",
    "DuplicateRows",
    "FeatureNoise",
    "MissingCells",
    "PermutationEstimate",
    "RelationalStudyResult",
    "StarSchema",
    "StudyResult",
    "TargetNoise",
    "attribute_quality_debt",
    "banzhaf_values",
    "coalition_key",
    "concentration_radius",
    "default_corruptions",
    "default_relational_corruptions",
    "estimate_permutation_shapley",
    "evaluate_relational_surface",
    "evaluate_risk_surface",
    "generate_star_schema",
    "load_nyc_taxi_star_schema",
    "normalize_query_weights",
    "shapley_interactions",
    "shapley_values",
]

__version__ = "0.2.0"
