# Backtesting and risk validation

The backtesting utilities evaluate a capital series against realized losses after removing positions where either side is non-finite. This makes them compatible with the warm-up period of rolling estimators.

## Exceedances

An exceedance occurs when

\[
L_t>K_t,
\]

where $K_t$ is forecast capital.

```python
from capacities_ml_fin.risk import exceedance_indicator

hits = exceedance_indicator(losses, capital)
```

Only aligned finite observations are returned.

## Capital summary

```python
from capacities_ml_fin.risk import capital_backtest

report = capital_backtest(losses, capital)
```

`CapitalBacktestResult` includes:

- number of evaluated observations;
- number and frequency of exceedances;
- average and standard deviation of capital;
- mean absolute change in capital;
- mean, maximum and total exceedance severity.

Exceedance severity is

\[
\max(L_t-K_t,0).
\]

## Stress-period backtest

```python
from capacities_ml_fin.risk import stress_backtest

stress_report = stress_backtest(
    losses,
    capital,
    stress_mask,
)
```

The function applies the same summary only to rows selected by a user-provided Boolean mask. The definition of “stress” is deliberately left to the empirical study.

## Kupiec unconditional coverage

```python
from capacities_ml_fin.risk import kupiec_coverage_test

result = kupiec_coverage_test(
    losses,
    capital,
    alpha=0.95,
)
```

The null exceedance probability is

\[
1-\alpha.
\]

The result contains the likelihood-ratio statistic, chi-squared p-value and one degree of freedom.

A high p-value does not prove that the model is correct; it indicates that the observed unconditional exceedance rate is not strongly inconsistent with the nominal rate under this test.

## Christoffersen first-order independence

```python
from capacities_ml_fin.risk import christoffersen_independence_test

independence = christoffersen_independence_test(losses, capital)
```

The test compares a common hit probability with a first-order Markov alternative based on the four hit transitions $00$, $01$, $10$, and $11$.

It tests clustering/first-order dependence of exceedances, not unconditional coverage itself.

## Diversification benefit

```python
from capacities_ml_fin.risk import diversification_benefit

benefit = diversification_benefit(
    losses_a,
    losses_b,
    risk_measure,
)
```

The reported quantity is

\[
\rho(A)+\rho(B)-\rho(A+B).
\]

A positive value indicates a diversification benefit under the supplied risk measure.

## Block-bootstrap confidence intervals

```python
from capacities_ml_fin.risk import block_bootstrap_interval

lower, upper = block_bootstrap_interval(
    values,
    block_size=20,
    n_resamples=2000,
    confidence_level=0.95,
    random_state=0,
)
```

The implementation uses a circular block bootstrap. Blocks preserve local serial dependence better than i.i.d. resampling.

## HAC mean interval

```python
from capacities_ml_fin.risk import hac_mean_interval

result = hac_mean_interval(values, max_lags=5)
```

The estimator uses a Newey-West/Bartlett long-run variance calculation and returns a normal-approximation interval for the mean.

If `max_lags=None`, the implementation uses its built-in sample-size rule.

## Numerical axiom checks

`check_risk_measure_axioms` evaluates several identities on two supplied finite loss vectors:

```python
from capacities_ml_fin.risk import check_risk_measure_axioms

report = check_risk_measure_axioms(
    risk_measure,
    first_losses,
    second_losses,
)
```

The `RiskAxiomReport` checks:

- monotonicity;
- cash/translation invariance;
- positive homogeneity;
- subadditivity;
- convexity;
- comonotonic additivity when the supplied pair is comonotonic.

These are **numerical checks on the supplied vectors**, not mathematical proofs that a callable satisfies an axiom over its entire domain.

## Comonotonicity

```python
from capacities_ml_fin.risk import is_comonotonic

is_comonotonic(x, y)
```

checks whether

\[
(x_i-x_j)(y_i-y_j)\ge0
\]

for all scenario pairs.

## Recommended backtesting sequence

For a VaR-style capital series, a compact evaluation sequence is:

```python
summary = capital_backtest(losses, capital)
coverage = kupiec_coverage_test(losses, capital, alpha=0.95)
independence = christoffersen_independence_test(losses, capital)
stress = stress_backtest(losses, capital, stress_mask)
```

Coverage and independence answer different questions and should generally be reported separately from capital level/stability and exceedance severity.

## API

See [Risk API — backtesting and validation](../api/risk.md#backtesting-and-validation).
