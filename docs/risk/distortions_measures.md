# Distortions and risk measures

A distortion changes the capacity assigned to an event before the Choquet integral is evaluated.

## Distortion contract

A valid distortion $g$ must map $[0,1]$ to $[0,1]$, be non-decreasing, and satisfy

\[
g(0)=0,
\qquad
g(1)=1.
\]

All distortion classes derive from `Distortion`.

```python
from capacities_ml_fin.risk import ExpectedShortfallDistortion

es = ExpectedShortfallDistortion(alpha=0.95)
es.validate()
es.is_concave()
```

`validate()` evaluates the public numerical contract on a grid. `is_concave()` and `is_convex()` use sampled second differences and should be understood as numerical checks of the supplied callable, not symbolic proofs.

## Identity distortion

\[
g(p)=p.
\]

```python
from capacities_ml_fin.risk import IdentityDistortion
```

Applied to a probability capacity, the resulting Choquet risk reduces to the corresponding weighted expectation over finite scenarios.

## VaR distortion

`ValueAtRiskDistortion(alpha)` implements the step

\[
g_\alpha(p)
=
\mathbf 1\{p>1-\alpha\}.
\]

```python
from capacities_ml_fin.risk import ValueAtRiskDistortion

var_distortion = ValueAtRiskDistortion(0.95)
```

It is associated with the lower empirical VaR convention used by the package.

## Expected Shortfall distortion

`ExpectedShortfallDistortion(alpha)` implements

\[
g_\alpha(p)
=
\min\left(\frac{p}{1-\alpha},1\right).
\]

This distortion is concave.

```python
from capacities_ml_fin.risk import ExpectedShortfallDistortion

es_distortion = ExpectedShortfallDistortion(0.95)
```

## Proportional-hazards distortion

```python
from capacities_ml_fin.risk import ProportionalHazardsDistortion

g = ProportionalHazardsDistortion(gamma=0.7)
```

implements

\[
g(p)=p^\gamma,
\qquad\gamma>0.
\]

For $0<\gamma<1$ it is concave; for $\gamma>1$ it is convex.

## Piecewise-linear and custom distortions

```python
from capacities_ml_fin.risk import PiecewiseLinearDistortion

piecewise = PiecewiseLinearDistortion(
    probabilities=(0.0, 0.05, 0.20, 1.0),
    values=(0.0, 0.20, 0.50, 1.0),
)
```

The constructor interpolates linearly and validates the resulting distortion.

A callable can be wrapped as:

```python
from capacities_ml_fin.risk import CustomDistortion

custom = CustomDistortion(
    lambda p: p ** 0.8,
    name="power_08",
)
```

The callable must preserve the input array shape and satisfy the distortion contract.

## Choquet risk measure

For a finite loss vector $L$ and event capacity $\mu$:

```python
from capacities_ml_fin.risk import choquet_risk_measure

rho = choquet_risk_measure(losses, capacity)
```

uses the generic discrete Choquet integral.

## Distortion risk measure

```python
from capacities_ml_fin.risk import distortion_risk_measure

rho = distortion_risk_measure(
    losses,
    es_distortion,
    sample_weight=weights,
)
```

If `capacity` is supplied instead of sample weights, the function evaluates

\[
E_{g\circ\mu}[L].
\]

```python
rho = distortion_risk_measure(
    losses,
    es_distortion,
    capacity=ambiguity_capacity,
)
```

## Convenience VaR and Expected Shortfall

```python
from capacities_ml_fin.risk import value_at_risk, expected_shortfall

var_95 = value_at_risk(losses, 0.95)
es_95 = expected_shortfall(losses, 0.95)
```

`value_at_risk` uses the lower empirical quantile. `expected_shortfall` is implemented through its distortion representation rather than a separate ad-hoc tail averaging routine.

## Generalized VaR and Tail-VaR

```python
from capacities_ml_fin.risk import (
    generalized_value_at_risk,
    generalized_tail_value_at_risk,
)

gvar = generalized_value_at_risk(
    losses,
    0.95,
    ambiguity_capacity,
    quantile="lower",
)

gtvar = generalized_tail_value_at_risk(
    losses,
    0.95,
    ambiguity_capacity,
)
```

The generalized VaR is the lower or upper quantile of the capacity distribution. Generalized Tail-VaR is evaluated by applying the Expected-Shortfall distortion to the supplied event capacity.

## Risk contributions

```python
from capacities_ml_fin.risk import risk_contributions

parts = risk_contributions(losses, risk_capacity)
```

The returned DataFrame decomposes the finite Choquet risk measure by sorted support increments. Its `contribution` column sums to the Choquet risk value.

This is a decomposition of the **ordered loss support**, not an Euler asset-level risk-contribution calculation.

## Callable object form

`DistortionRiskMeasure` wraps a `Distortion` as a callable object suitable for `RollingRiskEstimator` and other higher-level interfaces.

## API

See [Risk API — distortions and measures](../api/risk.md#distortions).
