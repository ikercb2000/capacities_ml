# imports
from dataclasses import dataclass


# capital backtest result
@dataclass(frozen=True, slots=True)
class CapitalBacktestResult:
    """Summary statistics for a capital forecast series."""

    n_observations: int
    n_exceedances: int
    exceedance_frequency: float
    average_capital: float
    capital_standard_deviation: float
    mean_absolute_capital_change: float
    mean_exceedance: float
    maximum_exceedance: float
    total_exceedance: float


# likelihood test result
@dataclass(frozen=True, slots=True)
class LikelihoodRatioTest:
    """Likelihood-ratio statistic and chi-squared p-value."""

    statistic: float
    p_value: float
    degrees_of_freedom: int


# HAC mean result
@dataclass(frozen=True, slots=True)
class HacMeanResult:
    """Mean estimate with a Newey-West HAC confidence interval."""

    mean: float
    standard_error: float
    lower_bound: float
    upper_bound: float
    max_lags: int
