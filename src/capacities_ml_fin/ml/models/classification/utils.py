# imports
from __future__ import annotations

import numpy as np

# modules
from capacities_ml_fin.ml.models.utils import capacity_design
from capacities_ml_fin.ml.optimization.enums import CapacityRepresentation


# strictly positive lower bound for scale normalization
_MIN_FEATURE_SCALE = 1e-8


# unit interval validation
def validate_unit_interval(X: np.ndarray) -> np.ndarray:
    """Require the commensurable ``[0, 1]`` criteria used by the paper."""
    if np.any(X < 0.0) or np.any(X > 1.0):
        raise ValueError(
            "Choquet predictors must lie in [0, 1]. Use "
            "CapacityNormalizer before fitting or predicting."
        )
    return X


# normalized feature scales
def normalized_feature_scales(raw_scales: np.ndarray) -> np.ndarray:
    """Normalize non-negative scales so that their maximum equals one."""
    scales = np.asarray(raw_scales, dtype=float).reshape(-1)
    if scales.size < 1 or np.any(~np.isfinite(scales)):
        raise ValueError("feature scales must contain finite values.")
    if np.any(scales < 0.0):
        raise ValueError("feature scales must be non-negative.")
    maximum = float(np.max(scales))
    if maximum < _MIN_FEATURE_SCALE:
        raise ValueError("at least one feature scale must be positive.")
    return scales / maximum


# Choquet score evaluation
def choquet_scores(
    design: np.ndarray,
    capacity_parameters: np.ndarray,
) -> np.ndarray:
    """Evaluate Choquet scores from a design matrix and capacity parameters."""
    return design @ capacity_parameters


# threshold classification
def threshold_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous Choquet scores into binary predictions."""
    return (scores >= threshold).astype(int)


# linear classifier callback
def linear_classifier(
    parameters: np.ndarray,
    *,
    design: np.ndarray,
    capacity_slice: slice,
    threshold_slice: slice,
) -> np.ndarray:
    """Evaluate a parameter vector as a threshold Choquet classifier."""
    scores = choquet_scores(design, parameters[capacity_slice])
    return threshold_predictions(scores, parameters[threshold_slice][0])


# scaled linear classifier callback
def scaled_linear_classifier(
    parameters: np.ndarray,
    *,
    matrix: np.ndarray,
    parameter_masks: tuple[int, ...],
    representation: CapacityRepresentation,
    capacity_slice: slice,
    feature_scales_slice: slice,
    threshold_slice: slice,
) -> np.ndarray:
    """Evaluate a Choquet classifier with learned normalized input scales."""
    feature_scales = normalized_feature_scales(
        parameters[feature_scales_slice]
    )
    design = capacity_design(
        matrix * feature_scales,
        parameter_masks,
        representation,
    )
    return linear_classifier(
        parameters,
        design=design,
        capacity_slice=capacity_slice,
        threshold_slice=threshold_slice,
    )


# Choquistic link function
def apply_choquistic_link(
    scores: np.ndarray,
    *,
    gamma: float,
    beta: float,
) -> np.ndarray:
    """Transform Choquet utilities into the paper's Choquistic log-odds."""
    return gamma * (scores - beta)


# Choquistic optimizer callback
def choquistic_logits(
    parameters: np.ndarray,
    *,
    design: np.ndarray,
    capacity_slice: slice,
    gamma_slice: slice,
    beta_slice: slice,
) -> np.ndarray:
    """Evaluate Choquistic log-odds from a numerical parameter vector."""
    scores = choquet_scores(design, parameters[capacity_slice])
    return apply_choquistic_link(
        scores,
        gamma=float(parameters[gamma_slice][0]),
        beta=float(parameters[beta_slice][0]),
    )
