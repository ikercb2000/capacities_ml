# imports
import polars as pl

# modules
from capacities_ml.capacities.base import Capacity

# ordered choquet integral
class ChoquetIntegral:
    capacity: Capacity
    x: pl.d