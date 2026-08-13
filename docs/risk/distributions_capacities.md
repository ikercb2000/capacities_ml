# Loss distributions and event capacities

The risk module represents a finite empirical loss distribution as a loss vector plus an event capacity on the same scenario set.

## `ProbabilityCapacity`

```python
import numpy as np
from capacities_ml_fin.risk import ProbabilityCapacity

capacity = ProbabilityCapacity(
    weights=np.array([0.1, 0.2, 0.3, 0.4])
)
```

Weights are normalized by the constructor. For a Boolean event mask $A$,

\[
\mu(A)=\sum_{i\in A}w_i.
\]

Because it implements `BaseCapacity`, it can be passed directly to generic Choquet and validation utilities.

## `DistortedCapacity`

```python
from capacities_ml_fin.risk import (
    DistortedCapacity,
    ExpectedShortfallDistortion,
)

risk_capacity = DistortedCapacity(
    capacity,
    ExpectedShortfallDistortion(alpha=0.95),
)
```

For any event $A$:

\[
\nu(A)=g(\mu(A)).
\]

No assumption of additivity is imposed on the base capacity. This means the same distortion class can be applied to a probability capacity or to an ambiguity capacity.

## Probability envelopes

### Upper envelope

For priors $P_1,\ldots,P_m$ represented by rows of scenario weights:

\[
\overline\mu(A)
=
\max_j P_j(A).
\]

```python
from capacities_ml_fin.risk import UpperEnvelopeCapacity

upper = UpperEnvelopeCapacity(
    prior_weights=np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.10, 0.20, 0.30, 0.40],
        ]
    )
)
```

Each prior row is normalized independently.

### Lower envelope

```python
from capacities_ml_fin.risk import LowerEnvelopeCapacity

lower = LowerEnvelopeCapacity(prior_weights=priors)
```

implements

\[
\underline\mu(A)
=
\min_j P_j(A).
\]

These capacities provide a simple finite representation of model ambiguity.

## `EmpiricalLossDistribution`

```python
from capacities_ml_fin.risk import EmpiricalLossDistribution

sample = EmpiricalLossDistribution(losses)
```

If neither `sample_weight` nor `capacity` is provided, equal scenario weights are used.

Weighted empirical probability:

```python
sample = EmpiricalLossDistribution(
    losses,
    sample_weight=weights,
)
```

Non-additive event capacity:

```python
sample = EmpiricalLossDistribution(
    losses,
    capacity=upper,
)
```

Use either `sample_weight` or `capacity`, not both.

## Survival and generalized distribution

The survival function is evaluated as

\[
S_{L,\mu}(x)=\mu(L>x).
\]

```python
sample.survival(1.0)
sample.survival([0.0, 0.5, 1.0])
```

The generalized distribution is

\[
G_{L,\mu}(x)=1-\mu(L>x).
\]

```python
sample.distribution(1.0)
```

This definition is important for non-additive capacities: the implementation does not replace it by a potentially different expression involving $\mu(L\le x)$.

## Lower and upper quantiles

```python
q_lower = sample.lower_quantile(0.95)
q_upper = sample.upper_quantile(0.95)
```

The implemented definitions are

\[
r^-_{L,\mu}(p)
=
\inf\{x:G_{L,\mu}(x)\ge p\},
\]

and

\[
r^+_{L,\mu}(p)
=
\inf\{x:G_{L,\mu}(x)>p\}.
\]

In a finite empirical distribution they may differ at jumps, which is why the API exposes both explicitly.

## Event probabilities

Any Boolean scenario event can be queried directly:

```python
large_loss = losses > 2.0
sample.event_probability(large_loss)
```

The result is additive only if the underlying capacity is additive.

## Validation with generic capacity tools

Because risk capacities implement `BaseCapacity`, generic validation can be reused for small finite scenario sets:

```python
from capacities_ml_fin.base.capacities import (
    validate_capacity,
    is_concave_capacity,
)

validate_capacity(risk_capacity, max_elements=8)
is_concave_capacity(risk_capacity, max_elements=8)
```

For realistic historical samples with hundreds of scenarios, exhaustive $2^n$ validation is not computationally appropriate; validate the mathematical construction or smaller representative instances instead.

## API

See [Risk API — capacities and distributions](../api/risk.md#event-capacities).
