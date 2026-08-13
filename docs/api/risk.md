# Risk API

## Event capacities

### `ProbabilityCapacity`

::: capacities_ml_fin.risk.capacities.capacities.ProbabilityCapacity

### `DistortedCapacity`

::: capacities_ml_fin.risk.capacities.capacities.DistortedCapacity

### `UpperEnvelopeCapacity`

::: capacities_ml_fin.risk.capacities.capacities.UpperEnvelopeCapacity

### `LowerEnvelopeCapacity`

::: capacities_ml_fin.risk.capacities.capacities.LowerEnvelopeCapacity

## Empirical distribution

### `EmpiricalLossDistribution`

::: capacities_ml_fin.risk.distributions.distributions.EmpiricalLossDistribution

## Distortions

### `Distortion`

::: capacities_ml_fin.risk.distortions.distortions.Distortion

### `IdentityDistortion`

::: capacities_ml_fin.risk.distortions.distortions.IdentityDistortion

### `ValueAtRiskDistortion`

::: capacities_ml_fin.risk.distortions.distortions.ValueAtRiskDistortion

### `ExpectedShortfallDistortion`

::: capacities_ml_fin.risk.distortions.distortions.ExpectedShortfallDistortion

### `ProportionalHazardsDistortion`

::: capacities_ml_fin.risk.distortions.distortions.ProportionalHazardsDistortion

### `PiecewiseLinearDistortion`

::: capacities_ml_fin.risk.distortions.distortions.PiecewiseLinearDistortion

### `CustomDistortion`

::: capacities_ml_fin.risk.distortions.distortions.CustomDistortion

### `validate_distortion`

::: capacities_ml_fin.risk.distortions.utils.validate_distortion

## Risk-measure functions

### `choquet_risk_measure`

::: capacities_ml_fin.risk.measures.utils.choquet_risk_measure

### `distortion_risk_measure`

::: capacities_ml_fin.risk.measures.utils.distortion_risk_measure

### `value_at_risk`

::: capacities_ml_fin.risk.measures.utils.value_at_risk

### `expected_shortfall`

::: capacities_ml_fin.risk.measures.utils.expected_shortfall

### `generalized_value_at_risk`

::: capacities_ml_fin.risk.measures.utils.generalized_value_at_risk

### `generalized_tail_value_at_risk`

::: capacities_ml_fin.risk.measures.utils.generalized_tail_value_at_risk

### `risk_contributions`

::: capacities_ml_fin.risk.measures.utils.risk_contributions

### `DistortionRiskMeasure`

::: capacities_ml_fin.risk.measures.measures.DistortionRiskMeasure

## Spectral and Kusuoka risk

### `SpectralRiskMeasure`

::: capacities_ml_fin.risk.spectral.spectral.SpectralRiskMeasure

### `KusuokaRiskMeasure`

::: capacities_ml_fin.risk.spectral.spectral.KusuokaRiskMeasure

## Conditional and rolling risk

### `ResidualBootstrapDistribution`

::: capacities_ml_fin.risk.conditional.conditional.ResidualBootstrapDistribution

### `RollingRiskEstimator`

::: capacities_ml_fin.risk.rolling.rolling.RollingRiskEstimator

### `rolling_risk_estimates`

::: capacities_ml_fin.risk.rolling.utils.rolling_risk_estimates

## Backtesting and validation

### `CapitalBacktestResult`

::: capacities_ml_fin.risk.backtesting.backtesting.CapitalBacktestResult

### `LikelihoodRatioTest`

::: capacities_ml_fin.risk.backtesting.backtesting.LikelihoodRatioTest

### `HacMeanResult`

::: capacities_ml_fin.risk.backtesting.backtesting.HacMeanResult

### `exceedance_indicator`

::: capacities_ml_fin.risk.backtesting.utils.exceedance_indicator

### `capital_backtest`

::: capacities_ml_fin.risk.backtesting.utils.capital_backtest

### `stress_backtest`

::: capacities_ml_fin.risk.backtesting.utils.stress_backtest

### `diversification_benefit`

::: capacities_ml_fin.risk.backtesting.utils.diversification_benefit

### `kupiec_coverage_test`

::: capacities_ml_fin.risk.backtesting.utils.kupiec_coverage_test

### `christoffersen_independence_test`

::: capacities_ml_fin.risk.backtesting.utils.christoffersen_independence_test

### `block_bootstrap_interval`

::: capacities_ml_fin.risk.backtesting.utils.block_bootstrap_interval

### `hac_mean_interval`

::: capacities_ml_fin.risk.backtesting.utils.hac_mean_interval

### `RiskAxiomReport`

::: capacities_ml_fin.risk.validation.validation.RiskAxiomReport

### `is_comonotonic`

::: capacities_ml_fin.risk.validation.utils.is_comonotonic

### `check_risk_measure_axioms`

::: capacities_ml_fin.risk.validation.utils.check_risk_measure_axioms
