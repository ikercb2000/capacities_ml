# imports
from enum import Enum

# solver enum
class Solver(str, Enum):
    """Optimization backend supported by :class:`Optimizer`."""

    SCIPY = "scipy"
    PYMOO = "pymoo"
    CVXPY = "cvxpy"

# capacity shape enum
class CapacityShape(str, Enum):
    """Shape constraints that can be applied to a capacity."""

    GENERAL = "general"
    CONVEX = "convex"
    CONCAVE = "concave"

# capacity representation enum
class CapacityRepresentation(str, Enum):
    """Numerical representation used for capacity parameters."""

    VALUES = "values"
    MOBIUS = "mobius"


# optimization sense enum
class OptimizationSense(str, Enum):
    """Direction of a CVXPY optimization objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
