# Machine-learning model API

## `ChoquetRegressor`

Least-squares scikit-learn regressor with a learned capacity and separate intercept.

::: capacities_ml_fin.ml.models.regression.choquet.ChoquetRegressor

## `ScaledChoquetRegressor`

Scaled Choquet regression $c + qC_\mu(x)$ with a normalized monotone capacity,
a learned intercept, and configurable bounds on $q$.

::: capacities_ml_fin.ml.models.regression.choquet.ScaledChoquetRegressor

## `ChoquetClassifier`

Deterministic binary threshold classifier. Inputs must lie in $[0,1]$ and be oriented so larger values favor the positive class. It exposes `choquet_score()` and `decision_function()` but intentionally no `predict_proba()`.

::: capacities_ml_fin.ml.models.classification.linear.ChoquetClassifier

## `ChoquisticRegression`

Probabilistic binary Choquet classifier with latent utility $C_\mu(x)$ and logistic link $\gamma(C_\mu(x)-\beta)$.

::: capacities_ml_fin.ml.models.classification.choquistic.ChoquisticRegression


## `ChoquetAutoRegressor`

Native sktime univariate forecaster. It aggregates autoregressive lags through a capacity and supports optional exogenous regressors, recursive prediction intervals and updates.

::: capacities_ml_fin.ml.models.time_series.choquet.ChoquetAutoRegressor
