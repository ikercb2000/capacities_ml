# imports
from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml.capacities import BaseCapacity
from capacities_ml.risk.distortions import Distortion
from capacities_ml.risk.validation.validation import RiskAxiomReport


# event enumeration
def _all_event_values(capacity: BaseCapacity, max_scenarios: int) -> np.ndarray:
    if capacity.n_elements > max_scenarios:
        raise ValueError(
            f"Exact validation is limited to {max_scenarios} scenarios; "
            f"received {capacity.n_elements}."
        )
    values = np.empty(1 << capacity.n_elements, dtype=float)
    positions = np.arange(capacity.n_elements)
    for encoding in range(values.size):
        event = (encoding & (1 << positions)) != 0
        values[encoding] = capacity.event_value(event)
    return values


# event capacity validation
def validate_event_capacity(
    capacity: BaseCapacity,
    *,
    max_scenarios: int = 16,
    tolerance: float = 1e-10,
) -> None:
    """Check normalization and monotonicity exactly on a finite event space."""
    values = _all_event_values(capacity, max_scenarios)
    if not np.isclose(values[0], 0.0, atol=tolerance):
        raise ValueError("An event capacity must assign zero to the empty event.")
    if not np.isclose(values[-1], 1.0, atol=tolerance):
        raise ValueError("An event capacity must assign one to the full event.")
    for event in range(values.size):
        for scenario in range(capacity.n_elements):
            if event & (1 << scenario):
                continue
            superset = event | (1 << scenario)
            if values[event] > values[superset] + tolerance:
                raise ValueError("The event capacity is not monotone.")


# concave event capacity
def is_concave_event_capacity(
    capacity: BaseCapacity,
    *,
    max_scenarios: int = 8,
    tolerance: float = 1e-10,
) -> bool:
    """Check submodularity of an event capacity by exhaustive enumeration."""
    values = _all_event_values(capacity, max_scenarios)
    for first in range(values.size):
        for second in range(values.size):
            if values[first | second] + values[first & second] > (
                values[first] + values[second] + tolerance
            ):
                return False
    return True


# convex event capacity
def is_convex_event_capacity(
    capacity: BaseCapacity,
    *,
    max_scenarios: int = 8,
    tolerance: float = 1e-10,
) -> bool:
    """Check supermodularity of an event capacity by exhaustive enumeration."""
    values = _all_event_values(capacity, max_scenarios)
    for first in range(values.size):
        for second in range(values.size):
            if values[first | second] + values[first & second] < (
                values[first] + values[second] - tolerance
            ):
                return False
    return True


# comonotonicity
def is_comonotonic(first: ArrayLike, second: ArrayLike) -> bool:
    """Return whether two finite random losses are comonotonic."""
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("first and second must be aligned one-dimensional arrays.")
    x_differences = x[:, None] - x[None, :]
    y_differences = y[:, None] - y[None, :]
    return bool(np.all(x_differences * y_differences >= 0.0))


# risk axiom checks
def check_risk_measure_axioms(
    risk_measure: Callable[[ArrayLike], float],
    first: ArrayLike,
    second: ArrayLike,
    *,
    cash: float = 1.0,
    scale: float = 2.0,
    tolerance: float = 1e-8,
) -> RiskAxiomReport:
    """Check risk-measure identities on two supplied finite loss vectors."""
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size == 0:
        raise ValueError("first and second must be aligned non-empty vectors.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("first and second must be finite.")
    if not np.isfinite(cash):
        raise ValueError("cash must be finite.")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and non-negative.")

    rho_x = float(risk_measure(x))
    rho_y = float(risk_measure(y))
    lower = np.minimum(x, y)
    upper = np.maximum(x, y)
    monotonicity = risk_measure(lower) <= risk_measure(upper) + tolerance
    cash_invariance = np.isclose(
        risk_measure(x + cash),
        rho_x + cash,
        atol=tolerance,
    )
    positive_homogeneity = np.isclose(
        risk_measure(scale * x),
        scale * rho_x,
        atol=tolerance,
    )
    subadditivity = risk_measure(x + y) <= rho_x + rho_y + tolerance
    convexity = risk_measure(0.5 * (x + y)) <= 0.5 * (rho_x + rho_y) + tolerance
    comonotonic_additivity = None
    if is_comonotonic(x, y):
        comonotonic_additivity = bool(
            np.isclose(risk_measure(x + y), rho_x + rho_y, atol=tolerance)
        )
    return RiskAxiomReport(
        monotonicity=bool(monotonicity),
        cash_invariance=bool(cash_invariance),
        positive_homogeneity=bool(positive_homogeneity),
        subadditivity=bool(subadditivity),
        convexity=bool(convexity),
        comonotonic_additivity=comonotonic_additivity,
    )


# distortion validation
def validate_distortion(distortion: Distortion) -> None:
    """Validate a distortion through its public numerical contract."""
    if not isinstance(distortion, Distortion):
        raise TypeError("distortion must be a Distortion instance.")
    distortion.validate()
