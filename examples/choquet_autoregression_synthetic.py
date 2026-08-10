# imports
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# modules
from capacities_ml.models import ChoquetAutoRegressor
from capacities_ml.optimization import Solver


# time-series data
def build_data(n_observations: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Generate a stable series with two lags and one exogenous variable."""
    X = np.linspace(0.0, 1.0, n_observations).reshape(-1, 1)
    noise = 0.008 * np.sin(np.arange(n_observations))
    y = np.empty(n_observations)
    y[:2] = (0.20, 0.24)

    for t in range(2, n_observations):
        lag_aggregate = 0.70 * y[t - 1] + 0.30 * y[t - 2]
        y[t] = 0.10 + 0.60 * lag_aggregate + 0.20 * X[t, 0] + noise[t]

    return X, y


# Choquet autoregression example
def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    X, y = build_data()

    model = ChoquetAutoRegressor(
        lags=2,
        solver=Solver.SCIPY,
        solver_options={"options": {"maxiter": 2_000, "ftol": 1e-12}},
        enforce_stationarity=True,
    ).fit(y, X)

    fitted = model.predict_in_sample(dynamic=False)
    dynamic = model.predict_in_sample(dynamic=True)
    future_X = np.array([[1.05], [1.10], [1.15]])
    forecast, intervals = model.predict(
        n_periods=3,
        X=future_X,
        return_conf_int=True,
    )

    print("Choquet autoregression with exogenous data")
    print("The first column is time and the second is the exogenous X")
    print(np.column_stack((np.arange(y.size), X[:, 0])))
    print("Observed series y")
    print(y)
    print()
    print(f"Order: {model.order_}")
    print(f"Learned phi: {model.phi_:.6f}")
    print(f"Learned intercept: {model.intercept_:.6f}")
    print(f"Learned exogenous coefficients: {model.exogenous_coef_}")
    print(f"Residual variance: {model.sigma2_:.8f}")
    print(f"AIC: {model.aic():.4f}")
    print(f"BIC: {model.bic():.4f}")
    print("Learned lag capacity")
    print(model.capacity_.to_named_dict())
    print()
    print("First five one-step predictions: predicted, observed, residual")
    print(np.column_stack((fitted[:5], y[model.lags:model.lags + 5], model.resid()[:5])))
    print("Last five dynamic predictions")
    print(dynamic[-5:])
    print()
    print("Future exogenous observations X_future")
    print(future_X)
    print("Three-step recursive forecast")
    print(forecast)
    print("Approximate 95% confidence intervals")
    print(intervals)


if __name__ == "__main__":
    main()
