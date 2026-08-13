# imports
from dataclasses import dataclass


# risk axiom report
@dataclass(frozen=True, slots=True)
class RiskAxiomReport:
    """Numerical checks of the standard monetary-risk axioms."""

    monotonicity: bool
    cash_invariance: bool
    positive_homogeneity: bool
    subadditivity: bool
    convexity: bool
    comonotonic_additivity: bool | None
