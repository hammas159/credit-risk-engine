from .binning import BinnedFeature, bin_categorical, bin_numeric, quantile_edges
from .fairness import audit, group_metrics
from .scorecard import (
    Scorecard,
    brier_score,
    calibration_table,
    fit_logistic,
    gini,
)

__version__ = "0.1.0"

__all__ = [
    "BinnedFeature",
    "Scorecard",
    "audit",
    "bin_categorical",
    "bin_numeric",
    "brier_score",
    "calibration_table",
    "fit_logistic",
    "gini",
    "group_metrics",
    "quantile_edges",
]
