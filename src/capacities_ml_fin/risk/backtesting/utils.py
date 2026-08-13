# imports
from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlogy
from scipy.stats import chi2, norm

# modules
from capacities_ml_fin.risk.backtesting.backtesting import (
    CapitalBacktestResult,
    HacMeanResult,
    LikelihoodRatioTest,
)


# finite alignment
def _aligned(losses: ArrayLike, capital: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    loss_values = np.asarray(losses, dtype=float)
    capital_values = np.asarray(capital, dtype=float)
    if loss_values.ndim != 1 or capital_values.ndim != 1:
        raise ValueError("losses and capital must be one-dimensional.")
    if loss_values.shape != capital_values.shape:
        raise ValueError("losses and capital must have the same shape.")
    valid = np.isfinite(loss_values) & np.isfinite(capital_values)
    if not np.any(valid):
        raise ValueError("losses and capital have no aligned finite observations.")
    return loss_values[valid], capital_values[valid]


# exceedance indicators
def exceedance_indicator(losses: ArrayLike, capital: ArrayLike) -> np.ndarray:
    """Return whether each finite realized loss exceeds forecast capital."""
    loss_values, capital_values = _aligned(losses, capital)
    return loss_values > capital_values


# capital backtest
def capital_backtest(losses: ArrayLike, capital: ArrayLike) -> CapitalBacktestResult:
    """Summarize capital level, stability, and exceedance severity."""
    loss_values, capital_values = _aligned(losses, capital)
    excess = np.maximum(loss_values - capital_values, 0.0)
    indicators = excess > 0.0
    positive_excess = excess[indicators]
    mean_change = (
        float(np.mean(np.abs(np.diff(capital_values))))
        if capital_values.size > 1
        else 0.0
    )
    return CapitalBacktestResult(
        n_observations=int(loss_values.size),
        n_exceedances=int(np.sum(indicators)),
        exceedance_frequency=float(np.mean(indicators)),
        average_capital=float(np.mean(capital_values)),
        capital_standard_deviation=float(np.std(capital_values)),
        mean_absolute_capital_change=mean_change,
        mean_exceedance=(float(np.mean(positive_excess)) if positive_excess.size else 0.0),
        maximum_exceedance=(float(np.max(positive_excess)) if positive_excess.size else 0.0),
        total_exceedance=float(np.sum(positive_excess)),
    )


# stress backtest
def stress_backtest(
    losses: ArrayLike,
    capital: ArrayLike,
    stress_mask: ArrayLike,
) -> CapitalBacktestResult:
    """Evaluate capital on a user-specified stress-period mask."""
    loss_values = np.asarray(losses, dtype=float)
    capital_values = np.asarray(capital, dtype=float)
    mask = np.asarray(stress_mask, dtype=bool)
    if mask.shape != loss_values.shape or capital_values.shape != loss_values.shape:
        raise ValueError("stress_mask, losses, and capital must have the same shape.")
    return capital_backtest(loss_values[mask], capital_values[mask])


# diversification benefit
def diversification_benefit(
    losses_a: ArrayLike,
    losses_b: ArrayLike,
    risk_measure: Callable[[ArrayLike], float],
) -> float:
    """Return ``rho(A) + rho(B) - rho(A + B)``."""
    first = np.asarray(losses_a, dtype=float)
    second = np.asarray(losses_b, dtype=float)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("losses_a and losses_b must be aligned one-dimensional arrays.")
    return float(risk_measure(first) + risk_measure(second) - risk_measure(first + second))


# Kupiec test
def kupiec_coverage_test(
    losses: ArrayLike,
    capital: ArrayLike,
    alpha: float,
) -> LikelihoodRatioTest:
    """Test whether the unconditional exceedance rate equals ``1 - alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    indicators = exceedance_indicator(losses, capital)
    n = indicators.size
    x = int(np.sum(indicators))
    expected = 1.0 - alpha
    observed = x / n
    null_log_likelihood = xlogy(x, expected) + xlogy(n - x, 1.0 - expected)
    alternative_log_likelihood = xlogy(x, observed) + xlogy(n - x, 1.0 - observed)
    statistic = max(0.0, float(-2.0 * (null_log_likelihood - alternative_log_likelihood)))
    return LikelihoodRatioTest(statistic, float(chi2.sf(statistic, 1)), 1)


# Christoffersen test
def christoffersen_independence_test(
    losses: ArrayLike,
    capital: ArrayLike,
) -> LikelihoodRatioTest:
    """Test first-order independence of consecutive exceedances."""
    indicators = exceedance_indicator(losses, capital).astype(int)
    if indicators.size < 2:
        raise ValueError("At least two aligned observations are required.")
    previous = indicators[:-1]
    current = indicators[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    independent = xlogy(n01 + n11, pi) + xlogy(n00 + n10, 1.0 - pi)
    markov = (
        xlogy(n01, pi0)
        + xlogy(n00, 1.0 - pi0)
        + xlogy(n11, pi1)
        + xlogy(n10, 1.0 - pi1)
    )
    statistic = max(0.0, float(-2.0 * (independent - markov)))
    return LikelihoodRatioTest(statistic, float(chi2.sf(statistic, 1)), 1)


# block bootstrap interval
def block_bootstrap_interval(
    values: ArrayLike,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    block_size: int = 10,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Return a circular block-bootstrap confidence interval."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("values must be a non-empty finite one-dimensional array.")
    if not isinstance(block_size, int) or not 1 <= block_size <= sample.size:
        raise ValueError("block_size must lie between one and the sample size.")
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    generator = np.random.default_rng(random_state)
    number_of_blocks = int(np.ceil(sample.size / block_size))
    offsets = np.arange(block_size)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        starts = generator.integers(0, sample.size, size=number_of_blocks)
        indices = ((starts[:, None] + offsets) % sample.size).reshape(-1)[: sample.size]
        estimates[index] = statistic(sample[indices])
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


# HAC mean interval
def hac_mean_interval(
    values: ArrayLike,
    *,
    max_lags: int | None = None,
    confidence_level: float = 0.95,
) -> HacMeanResult:
    """Estimate a mean and Newey-West interval for serially dependent data."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 2 or not np.all(np.isfinite(sample)):
        raise ValueError("values must contain at least two finite observations.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    if max_lags is None:
        max_lags = int(np.floor(4.0 * (sample.size / 100.0) ** (2.0 / 9.0)))
    if not isinstance(max_lags, int) or not 0 <= max_lags < sample.size:
        raise ValueError("max_lags must lie between zero and n_observations - 1.")
    centered = sample - np.mean(sample)
    long_run_variance = float(np.dot(centered, centered) / sample.size)
    for lag in range(1, max_lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / sample.size)
        bartlett_weight = 1.0 - lag / (max_lags + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance
    standard_error = float(np.sqrt(max(long_run_variance, 0.0) / sample.size))
    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    mean = float(np.mean(sample))
    return HacMeanResult(
        mean=mean,
        standard_error=standard_error,
        lower_bound=mean - critical_value * standard_error,
        upper_bound=mean + critical_value * standard_error,
        max_lags=max_lags,
    )
