"""Capacity-based machine learning and financial risk tools."""

from capacities_ml_fin.ml.aggregation import (
    BinaryProbabilityAggregationResult,
    RegressionAggregationResult,
    aggregate_binary_probabilities,
    aggregate_regression_predictions,
)
from capacities_ml_fin.ml.models import (
    ChoquetAutoRegressor,
    ChoquetClassifier,
    ChoquetRegressor,
    ChoquisticRegression,
    FuzzyChoquetInputLayer,
    FuzzyChoquetNeuralAutoRegressor,
    FuzzyChoquetNeuralClassifier,
    FuzzyChoquetNeuralRegressor,
    ScaledChoquetRegressor,
)

__all__ = [
    "BinaryProbabilityAggregationResult",
    "RegressionAggregationResult",
    "aggregate_binary_probabilities",
    "aggregate_regression_predictions",
    "ChoquetAutoRegressor",
    "ChoquetClassifier",
    "ChoquetRegressor",
    "ChoquisticRegression",
    "FuzzyChoquetInputLayer",
    "FuzzyChoquetNeuralAutoRegressor",
    "FuzzyChoquetNeuralClassifier",
    "FuzzyChoquetNeuralRegressor",
    "ScaledChoquetRegressor",
]
