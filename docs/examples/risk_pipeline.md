# Example: non-additive risk pipeline

This example combines weighted empirical distributions, distorted capacities, model ambiguity, spectral measures, rolling capital and backtesting.

## 1. Weighted empirical distribution

```python
import numpy as np

from capacities_ml_fin.risk import (
    EmpiricalLossDistribution,
    ProbabilityCapacity,
)

calibration_window = 256
weights = np.zeros(train_losses.size)
weights[-calibration_window:] = 1.0 / calibration_window

base_capacity = ProbabilityCapacity(weights)
distribution = EmpiricalLossDistribution(
    train_losses,
    capacity=base_capacity,
)

print(distribution.survival(1.0))
print(distribution.lower_quantile(0.95))
print(distribution.upper_quantile(0.95))
```

## 2. Distort the event capacity

```python
from capacities_ml_fin.risk import (
    DistortedCapacity,
    ExpectedShortfallDistortion,
    choquet_risk_measure,
    distortion_risk_measure,
)

es_distortion = ExpectedShortfallDistortion(0.95)
risk_capacity = DistortedCapacity(base_capacity, es_distortion)

rho_a = distortion_risk_measure(
    train_losses,
    es_distortion,
    capacity=base_capacity,
)
rho_b = choquet_risk_measure(train_losses, risk_capacity)

assert np.isclose(rho_a, rho_b)
```

The equality is a useful implementation check: one path composes the distortion inside the convenience function, the other constructs the distorted capacity explicitly.

## 3. Model ambiguity through an upper envelope

```python
from capacities_ml_fin.risk import UpperEnvelopeCapacity

short_weights = np.zeros(train_losses.size)
short_weights[-128:] = 1.0 / 128.0

ambiguity = UpperEnvelopeCapacity(
    np.vstack([weights, short_weights])
)
```

The event capacity now returns the largest event probability among the two finite prior weighting schemes.

## 4. Generalized and spectral risk

```python
from capacities_ml_fin.risk import (
    KusuokaRiskMeasure,
    SpectralRiskMeasure,
    generalized_tail_value_at_risk,
    generalized_value_at_risk,
)

gvar = generalized_value_at_risk(
    train_losses,
    0.95,
    ambiguity,
)

gtvar = generalized_tail_value_at_risk(
    train_losses,
    0.95,
    ambiguity,
)

spectral = SpectralRiskMeasure(
    levels=(0.90, 0.95, 0.99),
    weights=(0.20, 0.50, 0.30),
    quantile="lower",
)

kusuoka = KusuokaRiskMeasure(
    levels=(0.90, 0.95),
    weights=(0.40, 0.60),
    worst_case_weight=0.10,
)

spectral_value = spectral(train_losses, capacity=ambiguity)
kusuoka_value = kusuoka(train_losses, capacity=ambiguity)
```

## 5. One-step-ahead rolling capital

```python
from capacities_ml_fin.risk import (
    ValueAtRiskDistortion,
    rolling_risk_estimates,
)

capital = rolling_risk_estimates(
    losses,
    {
        "VaR 95%": ValueAtRiskDistortion(0.95),
        "ES 95%": ExpectedShortfallDistortion(0.95),
    },
    window=128,
    min_periods=128,
)
```

At each date, the capital uses only prior historical losses.

## 6. Backtest the forecasts

```python
from capacities_ml_fin.risk import (
    capital_backtest,
    christoffersen_independence_test,
    kupiec_coverage_test,
)

var_capital = capital["VaR 95%"]

summary = capital_backtest(losses, var_capital)
coverage = kupiec_coverage_test(losses, var_capital, alpha=0.95)
independence = christoffersen_independence_test(losses, var_capital)
```

Report the three outputs separately: the descriptive capital/exceedance summary, unconditional hit-rate calibration, and hit independence test answer different questions.
