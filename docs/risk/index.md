# Risk measurement

The risk module applies capacities to **finite loss scenarios**. The sign convention throughout the module is:

\[
\text{larger value} = \text{larger loss}.
\]

It combines the capacity abstraction from `base` with empirical distributions, probability distortions, generalized quantiles, spectral aggregation, conditional scenarios, rolling capital forecasts and backtesting utilities.

## Conceptual pipeline

```text
finite losses
    ↓
EmpiricalLossDistribution
    ↓
base event capacity
    ├── ProbabilityCapacity
    ├── UpperEnvelopeCapacity
    └── LowerEnvelopeCapacity
    ↓
optional distortion g
    ↓
DistortedCapacity = g ∘ μ
    ↓
Choquet risk measure / generalized quantiles
    ↓
rolling estimator
    ↓
capital backtest
```

## Finite event capacities

A loss vector of length $n$ defines $n$ scenarios. An event is a Boolean mask over those scenarios. `ProbabilityCapacity` evaluates events additively from normalized scenario weights, while envelope capacities evaluate the maximum or minimum probability over a finite collection of priors.

[Distributions & capacities](distributions_capacities.md)

## Distortions

A distortion is an increasing map

\[
g:[0,1]\to[0,1],
\qquad g(0)=0,
\quad g(1)=1.
\]

`DistortedCapacity` composes it with any `BaseCapacity`:

\[
\nu(A)=g(\mu(A)).
\]

The package provides identity, VaR, Expected Shortfall, proportional-hazards, piecewise-linear and custom distortions.

[Distortions & measures](distortions_measures.md)

## Generalized distributions and quantiles

For losses $L$ and event capacity $\mu$, `EmpiricalLossDistribution` defines

\[
G_{L,\mu}(x)
=
1-\mu(L>x).
\]

The lower and upper quantiles are then implemented directly from this generalized distribution.

This lets the same API work with ordinary probability weights or genuinely non-additive event capacities.

## Spectral and Kusuoka-style measures

`SpectralRiskMeasure` forms a finite normalized mixture of generalized quantiles. `KusuokaRiskMeasure` combines generalized Tail-VaR levels with an optional worst-case component.

[Spectral & Kusuoka](spectral_kusuoka.md)

## Conditional and rolling risk

`ResidualBootstrapDistribution` converts a point predictor into empirical conditional scenario distributions. `RollingRiskEstimator` turns any supported distortion or callable measure into one-step-ahead rolling/expanding capital forecasts without using the contemporaneous loss.

[Conditional & rolling](rolling_conditional.md)

## Backtesting

The module includes exceedance summaries, Kupiec unconditional coverage, Christoffersen first-order independence, stress subsets, block-bootstrap intervals and Newey-West HAC mean intervals.

[Backtesting & validation](backtesting_validation.md)

## A compact risk workflow

```python
from capacities_ml_fin.risk import (
    ExpectedShortfallDistortion,
    RollingRiskEstimator,
    capital_backtest,
    kupiec_coverage_test,
)

alpha = 0.95
estimator = RollingRiskEstimator(
    ExpectedShortfallDistortion(alpha),
    window=250,
    min_periods=250,
).fit(losses)

capital = estimator.predict_in_sample()
summary = capital_backtest(losses, capital)
coverage = kupiec_coverage_test(losses, capital, alpha=alpha)
```

The finite-observation filtering performed by the backtesting utilities makes the initial warm-up `NaN` values in rolling capital series compatible with later evaluation.
