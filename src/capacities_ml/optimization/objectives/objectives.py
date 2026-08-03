# imports
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

# modules

# objective aliases
FloatArray = NDArray[np.float64]
Predictor = Any


# objective specification base class
class ObjectiveSpec:
    """Callable numerical objective with an optional symbolic translation."""

    def __call__(self, parameters: FloatArray) -> float:
        raise NotImplementedError


# squared error objective specification
@dataclass(slots=True)
class SquaredErrorObjective(ObjectiveSpec):
    target: ArrayLike
    predictor: Predictor
    penalty: Any = None
    mean: bool = True
    symbolic_predictor: Any = None

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=float).reshape(-1)

    def __call__(self, parameters: FloatArray) -> float:
        prediction = np.asarray(self.predictor(parameters), dtype=float).reshape(-1)
        if prediction.shape != self.target.shape:
            raise ValueError("predictor output and target have incompatible shapes.")
        residual = self.target - prediction
        loss = np.mean(residual**2) if self.mean else np.sum(residual**2)
        return float(loss + (0.0 if self.penalty is None else self.penalty(parameters)))


# absolute error objective specification
@dataclass(slots=True)
class AbsoluteErrorObjective(ObjectiveSpec):
    target: ArrayLike
    predictor: Predictor
    penalty: Any = None
    mean: bool = True
    symbolic_predictor: Any = None

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=float).reshape(-1)

    def __call__(self, parameters: FloatArray) -> float:
        prediction = np.asarray(self.predictor(parameters), dtype=float).reshape(-1)
        if prediction.shape != self.target.shape:
            raise ValueError("predictor output and target have incompatible shapes.")
        residual = np.abs(self.target - prediction)
        loss = np.mean(residual) if self.mean else np.sum(residual)
        return float(loss + (0.0 if self.penalty is None else self.penalty(parameters)))


# quantile loss objective specification
@dataclass(slots=True)
class QuantileLossObjective(ObjectiveSpec):
    target: ArrayLike
    predictor: Predictor
    quantile: float
    penalty: Any = None
    mean: bool = True
    symbolic_predictor: Any = None

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must lie in (0, 1).")
        self.target = np.asarray(self.target, dtype=float).reshape(-1)

    def __call__(self, parameters: FloatArray) -> float:
        prediction = np.asarray(self.predictor(parameters), dtype=float).reshape(-1)
        if prediction.shape != self.target.shape:
            raise ValueError("predictor output and target have incompatible shapes.")
        residual = self.target - prediction
        losses = np.maximum(self.quantile * residual, (self.quantile - 1.0) * residual)
        loss = np.mean(losses) if self.mean else np.sum(losses)
        return float(loss + (0.0 if self.penalty is None else self.penalty(parameters)))


# logistic negative log-likelihood objective specification
@dataclass(slots=True)
class LogisticNegativeLogLikelihood(ObjectiveSpec):
    target: ArrayLike
    predictor: Predictor
    sample_weight: ArrayLike | None = None
    penalty: Any = None
    mean: bool = True
    symbolic_predictor: Any = None

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=float).reshape(-1)
        if np.any((self.target != 0.0) & (self.target != 1.0)):
            raise ValueError("Binary targets must contain only 0 and 1.")
        if self.sample_weight is None:
            self.sample_weight = np.ones_like(self.target)
        else:
            self.sample_weight = np.asarray(self.sample_weight, dtype=float).reshape(-1)
            if self.sample_weight.shape != self.target.shape or np.any(self.sample_weight < 0):
                raise ValueError("sample_weight must be non-negative and match target.")

    def __call__(self, parameters: FloatArray) -> float:
        logits = np.asarray(self.predictor(parameters), dtype=float).reshape(-1)
        if logits.shape != self.target.shape:
            raise ValueError("linear_predictor output and target are incompatible.")
        losses = self.sample_weight * (np.logaddexp(0.0, logits) - self.target * logits)
        loss = np.mean(losses) if self.mean else np.sum(losses)
        return float(loss + (0.0 if self.penalty is None else self.penalty(parameters)))


# zero-one loss objective specification
@dataclass(slots=True)
class ZeroOneLossObjective(ObjectiveSpec):
    target: ArrayLike
    predictor: Predictor
    mean: bool = False

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=int).reshape(-1)

    def __call__(self, parameters: FloatArray) -> float:
        prediction = np.asarray(self.predictor(parameters), dtype=int).reshape(-1)
        if prediction.shape != self.target.shape:
            raise ValueError("classifier output and target are incompatible.")
        errors = prediction != self.target
        return float(np.mean(errors) if self.mean else np.sum(errors))
