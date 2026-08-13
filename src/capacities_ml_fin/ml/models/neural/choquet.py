# imports
from __future__ import annotations

from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted, validate_data

# modules
from capacities_ml_fin.ml.models.utils import (
    capacity_design,
    fitted_universe,
    resolve_solver,
)
from capacities_ml_fin.ml.optimization import FullCapacity, Problem, Solver
from capacities_ml_fin.ml.optimization.backends.to_scipy import ScipyOptimizer
from capacities_ml_fin.ml.optimization.constraints import (
    LinearConstraintSystem,
    NonlinearConstraintSpec,
    VariableBounds,
)
from capacities_ml_fin.ml.optimization.parametrization import ParameterBlock, ParameterLayout
from capacities_ml_fin.ml.optimization.problem import OptimizationProblem
from capacities_ml_fin.ml.optimization.sparsity import CapacitySparsity


# activation function
def _activation(name: str, values: np.ndarray) -> np.ndarray:
    if name == "identity":
        return values
    if name == "logistic":
        return expit(values)
    if name == "tanh":
        return np.tanh(values)
    if name == "relu":
        return np.maximum(values, 0.0)
    raise ValueError("activation must be one of {'identity', 'logistic', 'tanh', 'relu'}.")


# activation derivative
def _activation_derivative(
    name: str,
    preactivation: np.ndarray,
    activated: np.ndarray,
) -> np.ndarray:
    if name == "identity":
        return np.ones_like(preactivation)
    if name == "logistic":
        return activated * (1.0 - activated)
    if name == "tanh":
        return 1.0 - activated**2
    if name == "relu":
        return (preactivation > 0.0).astype(float)
    raise ValueError("activation must be one of {'identity', 'logistic', 'tanh', 'relu'}.")


# shared Choquet neural implementation
class _ChoquetNeuralMixin:
    # estimator hyperparameter validation
    def _validate_hyperparameters(self) -> None:
        if not isinstance(self.n_hidden, (int, np.integer)) or self.n_hidden < 1:
            raise ValueError("n_hidden must be a positive integer.")
        _activation(self.activation, np.zeros((1, 1)))
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if float(self.tol) <= 0.0:
            raise ValueError("tol must be positive.")
        if float(self.alpha) < 0.0:
            raise ValueError("alpha must be non-negative.")
        self.solver_ = resolve_solver(self.solver)
        if self.solver_ is not Solver.SCIPY:
            raise ValueError("Choquet neural estimators currently support solver=Solver.SCIPY.")

    # constrained neural optimization problem
    def _make_problem(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        *,
        classification: bool,
    ) -> tuple[OptimizationProblem, Any, Any, Any]:
        self._validate_hyperparameters()

        # capacity parameterization shared by all hidden neurons
        sparsity = self.sparsity if self.sparsity is not None else FullCapacity()
        compilation = sparsity.compile(self.universe_.n_elements)
        bundle = compilation.bundle
        design = capacity_design(
            matrix,
            bundle.parameter_masks,
            bundle.representation,
        )
        n_capacity = bundle.n_parameters
        layout = ParameterLayout(
            ParameterBlock("capacities", self.n_hidden * n_capacity),
            ParameterBlock("hidden_bias", self.n_hidden),
            ParameterBlock("output_weight", self.n_hidden),
            ParameterBlock("output_bias", 1),
        )
        capacity_slice = layout.slice("capacities")
        hidden_bias_slice = layout.slice("hidden_bias")
        output_weight_slice = layout.slice("output_weight")
        output_bias_slice = layout.slice("output_bias")

        # feasible capacity initialization and random neural weights
        rng = np.random.default_rng(self.random_state)
        initial = layout.pack(
            {
                "capacities": np.tile(compilation.initial_parameters, self.n_hidden),
                "hidden_bias": rng.normal(0.0, 0.05, self.n_hidden),
                "output_weight": rng.normal(0.0, 1.0 / np.sqrt(self.n_hidden), self.n_hidden),
                "output_bias": np.array([0.0]),
            }
        )
        if classification:
            prevalence = np.clip(np.average(target, weights=sample_weight), 1e-6, 1.0 - 1e-6)
            initial[output_bias_slice] = np.log(prevalence / (1.0 - prevalence))
        else:
            initial[output_bias_slice] = np.average(target, weights=sample_weight)

        # forward pass through the Choquet hidden layer
        def forward(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            capacities = parameters[capacity_slice].reshape(self.n_hidden, n_capacity)
            preactivation = design @ capacities.T + parameters[hidden_bias_slice]
            hidden = _activation(self.activation, preactivation)
            output = hidden @ parameters[output_weight_slice] + parameters[output_bias_slice][0]
            return output, preactivation, hidden

        weight_sum = float(np.sum(sample_weight))

        # regression or logistic training loss
        def objective(parameters: np.ndarray) -> float:
            output, _, _ = forward(parameters)
            if classification:
                losses = np.logaddexp(0.0, output) - target * output
            else:
                losses = (output - target) ** 2
            regularization = self.alpha * np.sum(parameters[output_weight_slice] ** 2)
            return float(np.sum(sample_weight * losses) / weight_sum + regularization)

        # analytical gradient of the neural loss
        def gradient(parameters: np.ndarray) -> np.ndarray:
            output, preactivation, hidden = forward(parameters)
            output_weights = parameters[output_weight_slice]
            if classification:
                output_gradient = sample_weight * (expit(output) - target) / weight_sum
            else:
                output_gradient = 2.0 * sample_weight * (output - target) / weight_sum
            hidden_gradient = (
                output_gradient[:, None]
                * output_weights[None, :]
                * _activation_derivative(self.activation, preactivation, hidden)
            )
            result = np.zeros_like(parameters)
            result[capacity_slice] = (hidden_gradient.T @ design).reshape(-1)
            result[hidden_bias_slice] = np.sum(hidden_gradient, axis=0)
            result[output_weight_slice] = (
                hidden.T @ output_gradient + 2.0 * self.alpha * output_weights
            )
            result[output_bias_slice] = np.sum(output_gradient)
            return result

        # repeat the capacity constraints for every hidden neuron
        lower = layout.bounds().lower
        upper = layout.bounds().upper
        linear_constraints: list[LinearConstraintSystem] = []
        nonlinear_constraints: list[NonlinearConstraintSpec] = []
        for unit in range(self.n_hidden):
            start = capacity_slice.start + unit * n_capacity
            stop = start + n_capacity
            lower[start:stop] = bundle.constraints.bounds.lower
            upper[start:stop] = bundle.constraints.bounds.upper
            linear_constraints.extend(
                constraint.embed(start=start, total_parameters=layout.n_parameters)
                for constraint in bundle.constraints.linear_constraints
            )
            for constraint in bundle.constraints.nonlinear_constraints:
                nonlinear_constraints.append(
                    NonlinearConstraintSpec(
                        function=(
                            lambda parameters,
                            constraint=constraint,
                            start=start,
                            stop=stop: constraint.values(parameters[start:stop])
                        ),
                        lower=constraint.lower,
                        upper=constraint.upper,
                        jacobian=None,
                        name=f"hidden_{unit}_{constraint.name}",
                    )
                )

        problem = OptimizationProblem(
            objective=objective,
            initial_parameters=initial,
            bounds=VariableBounds(lower, upper),
            linear_constraints=tuple(linear_constraints),
            nonlinear_constraints=tuple(nonlinear_constraints),
            gradient=gradient,
            layout=layout,
            name=("choquet_neural_classifier" if classification else "choquet_neural_regressor"),
        )
        return problem, compilation, design, forward

    # common fitting routine
    def _fit_core(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        *,
        classification: bool,
    ) -> None:
        problem, compilation, _, forward = self._make_problem(
            matrix,
            target,
            sample_weight,
            classification=classification,
        )
        options: dict[str, Any] = {
            "method": "SLSQP",
            "tolerance": float(self.tol),
            "options": {"maxiter": int(self.max_iter)},
        }
        if self.solver_options is not None:
            custom_options = dict(self.solver_options)
            scipy_minimize_options = dict(options["options"])
            scipy_minimize_options.update(custom_options.pop("options", {}))
            options.update(custom_options)
            options["options"] = scipy_minimize_options

        # constrained SLSQP optimization
        result = ScipyOptimizer(**options).solve(problem)
        if not result.success:
            warn(
                f"Choquet neural optimization did not converge: {result.message}",
                ConvergenceWarning,
                stacklevel=2,
            )

        # decode each learned hidden capacity
        n_capacity = compilation.bundle.n_parameters
        capacity_parameters = problem.layout.unpack(result.parameters)["capacities"].reshape(
            self.n_hidden, n_capacity
        )
        decoder = Problem.from_capacity(
            universe=self.universe_,
            objective=lambda parameters: 0.0,
            sparsity=self.sparsity if self.sparsity is not None else FullCapacity(),
        )
        self.capacities_ = tuple(decoder.decode(values) for values in capacity_parameters)
        self.capacity_parameters_ = capacity_parameters
        self.problem_ = problem
        self.result_ = result
        self.parameterization_ = compilation.bundle
        self.sparsity_ = self.sparsity if self.sparsity is not None else FullCapacity()
        self.loss_ = float(result.objective_value)
        self.n_iter_ = result.n_iterations
        self._forward_training = forward

    # raw network output
    def _raw_predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, ["result_", "parameterization_"])
        matrix = validate_data(self, X, reset=False, dtype=float)
        design = capacity_design(
            matrix,
            self.parameterization_.parameter_masks,
            self.parameterization_.representation,
        )
        blocks = self.problem_.layout.unpack(self.result_.parameters)
        preactivation = design @ self.capacity_parameters_.T + blocks["hidden_bias"]
        hidden = _activation(self.activation, preactivation)
        return hidden @ blocks["output_weight"] + blocks["output_bias"][0]

    # feature names
    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        """Return the input names associated with every learned capacity."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        return np.asarray(self.universe_.var_names, dtype=object)


# Choquet neural regressor
class ChoquetNeuralRegressor(_ChoquetNeuralMixin, RegressorMixin, BaseEstimator):
    """One-hidden-layer regressor with Choquet aggregation units.

    Each hidden unit learns its own normalized monotone capacity. Activated
    hidden outputs are combined by a linear output layer and optimized jointly.

    Parameters
    ----------
    n_hidden : int, default=4
        Number of Choquet hidden units.
    activation : {"identity", "tanh", "relu", "logistic"}, default="tanh"
        Hidden activation function.
    sparsity : CapacitySparsity, optional
        Capacity parameterization shared by the hidden units.
    alpha : float, default=1e-4
        L2 regularization strength.
    max_iter : int, default=300
        Maximum optimization iterations.
    tol : float, default=1e-6
        Optimization tolerance.
    random_state : int, optional
        Seed controlling initialization.
    solver : {"scipy", "cvxpy", "pymoo"} or Solver, default="scipy"
        Optimization backend.
    solver_options : dict, optional
        Additional backend options.

    Attributes
    ----------
    capacities_ : list of BaseCapacity
        Capacity learned by each hidden unit.
    output_weights_ : ndarray of shape (n_hidden,)
        Linear output weights.
    intercept_ : float
        Output intercept.
    result_ : OptimizationResult
        Numerical optimizer result.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    """

    def __init__(
        self,
        n_hidden: int = 4,
        activation: str = "tanh",
        sparsity: CapacitySparsity | None = None,
        alpha: float = 0.0001,
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: int | None = None,
        solver: Solver | str = "scipy",
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        self.n_hidden = n_hidden
        self.activation = activation
        self.sparsity = sparsity
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.solver = solver
        self.solver_options = solver_options

    def __sklearn_tags__(self):
        """Declare stochastic behavior when no seed is supplied."""
        tags = super().__sklearn_tags__()
        tags.non_deterministic = self.random_state is None
        return tags

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "ChoquetNeuralRegressor":
        # regression target validation
        matrix, target = validate_data(
            self,
            X,
            y,
            dtype=float,
            ensure_2d=True,
            ensure_min_samples=1,
        )
        self.universe_ = fitted_universe(self)
        target = np.asarray(target, dtype=float)
        weights = _sample_weights(sample_weight, target.size)
        self._fit_core(matrix, target, weights, classification=False)
        return self

    # continuous prediction
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous responses."""
        return self._raw_predict(X)


# Choquet neural classifier
class ChoquetNeuralClassifier(_ChoquetNeuralMixin, ClassifierMixin, BaseEstimator):
    """Binary classifier with Choquet hidden units and logistic output.

    Parameters
    ----------
    n_hidden : int, default=4
        Number of Choquet hidden units.
    activation : {"identity", "tanh", "relu", "logistic"}, default="tanh"
        Hidden activation function.
    sparsity : CapacitySparsity, optional
        Capacity parameterization shared by hidden units.
    alpha : float, default=1e-4
        L2 regularization strength.
    max_iter : int, default=300
        Maximum optimization iterations.
    tol : float, default=1e-6
        Optimization tolerance.
    random_state : int, optional
        Seed controlling initialization.
    solver : {"scipy", "cvxpy", "pymoo"} or Solver, default="scipy"
        Optimization backend.
    solver_options : dict, optional
        Additional backend options.
    class_weight : dict, "balanced", or None, default=None
        Per-class likelihood weights.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Original binary labels.
    capacities_ : list of BaseCapacity
        Capacity learned by every hidden unit.
    output_weights_ : ndarray of shape (n_hidden,)
        Learned logit weights.
    intercept_ : float
        Learned logit intercept.
    result_ : OptimizationResult
        Numerical optimizer result.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    """

    def __init__(
        self,
        n_hidden: int = 4,
        activation: str = "tanh",
        sparsity: CapacitySparsity | None = None,
        alpha: float = 0.0001,
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: int | None = None,
        solver: Solver | str = "scipy",
        solver_options: dict[str, Any] | None = None,
        class_weight: dict[Any, float] | str | None = None,
    ) -> None:
        self.n_hidden = n_hidden
        self.activation = activation
        self.sparsity = sparsity
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.solver = solver
        self.solver_options = solver_options
        self.class_weight = class_weight

    def __sklearn_tags__(self):
        """Declare binary output and optional stochastic behavior."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.non_deterministic = self.random_state is None
        return tags

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "ChoquetNeuralClassifier":
        # input and binary-label validation
        matrix, raw_target = validate_data(
            self,
            X,
            y,
            dtype=float,
            ensure_2d=True,
            ensure_min_samples=1,
        )
        self.universe_ = fitted_universe(self)
        encoder = LabelEncoder()
        target = encoder.fit_transform(raw_target)
        if encoder.classes_.size != 2:
            raise ValueError("y must contain exactly two distinct labels.")
        weights = compute_sample_weight(self.class_weight, raw_target).astype(float)
        weights *= _sample_weights(sample_weight, target.size)
        if not np.any(weights > 0.0):
            raise ValueError("At least one sample must have positive weight.")
        self._fit_core(matrix, target.astype(float), weights, classification=True)
        self.classes_ = encoder.classes_
        return self

    # output logits
    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return logits produced by the output neuron."""
        return self._raw_predict(X)

    # class probabilities
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return probabilities for both classes."""
        positive = expit(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    # log class probabilities
    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return numerically stable class log-probabilities."""
        logits = self.decision_function(X)
        return np.column_stack((-np.logaddexp(0.0, logits), -np.logaddexp(0.0, -logits)))

    # binary prediction
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict binary labels."""
        check_is_fitted(self, ["classes_", "result_", "parameterization_"])
        return self.classes_[(self.decision_function(X) >= 0.0).astype(int)]


# sample-weight validation
def _sample_weights(sample_weight: ArrayLike | None, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=float)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.shape != (n_samples,):
        raise ValueError("sample_weight must contain one value per observation.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample_weight must be finite and non-negative.")
    if not np.any(weights > 0.0):
        raise ValueError("At least one sample must have positive weight.")
    return weights
