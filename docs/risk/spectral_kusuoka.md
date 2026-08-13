# Spectral and Kusuoka-style risk measures

The spectral layer works directly with generalized quantiles produced by `EmpiricalLossDistribution`.

## Finite spectral risk measure

`SpectralRiskMeasure` takes quantile levels and non-negative weights. The weights are normalized internally to sum to one.

```python
from capacities_ml_fin.risk import SpectralRiskMeasure

spectral = SpectralRiskMeasure(
    levels=(0.90, 0.95, 0.99),
    weights=(0.20, 0.50, 0.30),
    quantile="upper",
)

rho = spectral(losses)
```

For levels $p_j$ and normalized weights $w_j$:

\[
\rho(L)
=
\sum_j w_j r_{L,\mu}(p_j).
\]

The quantile can be lower or upper. If no capacity is supplied at call time, an empirical probability capacity is constructed from equal or supplied sample weights.

## Capacity-based spectral measure

```python
rho = spectral(
    losses,
    capacity=ambiguity_capacity,
)
```

Now each quantile is computed from

\[
G_{L,\mu}(x)=1-\mu(L>x)
\]

rather than from an ordinary empirical CDF.

This is what makes the object a finite generalized spectral measure rather than merely a weighted average of standard empirical quantiles.

## Kusuoka-style finite mixture

`KusuokaRiskMeasure` combines generalized Tail-VaR levels and an optional worst-case term.

```python
from capacities_ml_fin.risk import KusuokaRiskMeasure

kusuoka = KusuokaRiskMeasure(
    levels=(0.90, 0.95),
    weights=(0.40, 0.60),
    worst_case_weight=0.10,
)

rho = kusuoka(losses, capacity=ambiguity_capacity)
```

The implemented finite form is

\[
\rho(L)
=
\alpha\max_i L_i
+
(1-\alpha)
\sum_j w_j\,\operatorname{GTVaR}_{p_j}^{\mu}(L),
\]

where `worst_case_weight` is $\alpha$ and the supplied `weights` are normalized.

The object is a finite computational representation inspired by the capacity-based Kusuoka-type structure implemented elsewhere in the project; it is not an optimizer over an arbitrary infinite family of probability measures on confidence levels.

## Generalized Tail-VaR building block

For each level,

```python
generalized_tail_value_at_risk(
    losses,
    level,
    capacity,
)
```

evaluates the Expected-Shortfall distortion composed with the supplied capacity.

This makes the Kusuoka object consistent with the rest of the distortion-risk API rather than maintaining a separate quantile integration implementation.

## Comparing spectral and Kusuoka objects

`SpectralRiskMeasure` is appropriate when the desired object is explicitly a finite weighted mixture of selected quantiles.

`KusuokaRiskMeasure` is appropriate when the desired basis consists of generalized Tail-VaR values plus an optional worst-case component.

## Example under ambiguity

```python
import numpy as np
from capacities_ml_fin.risk import UpperEnvelopeCapacity

priors = np.vstack([weights_long_window, weights_short_window])
ambiguity = UpperEnvelopeCapacity(priors)

spectral_value = spectral(losses, capacity=ambiguity)
kusuoka_value = kusuoka(losses, capacity=ambiguity)
```

The same finite loss scenarios are being evaluated under a non-additive upper-envelope event capacity.

## API

See [Risk API — spectral measures](../api/risk.md#spectral-and-kusuoka-risk).
