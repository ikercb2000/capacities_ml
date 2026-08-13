# imports
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray

# modules
from capacities_ml_fin.ml.optimization.constraints import VariableBounds

# parametrization aliases
FloatArray = NDArray[np.float64]

# parameter block
@dataclass(frozen=True, slots=True)
class ParameterBlock:
    """Named contiguous block inside a global optimization vector."""

    name: str
    size: int
    lower: float | ArrayLike = -np.inf
    upper: float | ArrayLike = np.inf

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter block names cannot be empty.")
        if self.size < 1:
            raise ValueError("Parameter block size must be positive.")

    def bounds(self) -> VariableBounds:
        lower = np.broadcast_to(np.asarray(self.lower, dtype=float), (self.size,)).copy()
        upper = np.broadcast_to(np.asarray(self.upper, dtype=float), (self.size,)).copy()
        return VariableBounds(lower, upper)


# parameter layout
class ParameterLayout:
    """Maps semantic parameter names to slices of one numerical vector."""

    def __init__(self, *blocks: ParameterBlock) -> None:
        if not blocks:
            raise ValueError("At least one parameter block is required.")
        names = [block.name for block in blocks]
        if len(names) != len(set(names)):
            raise ValueError("Parameter block names must be unique.")
        self._blocks = tuple(blocks)
        self._slices: dict[str, slice] = {}
        cursor = 0
        for block in self._blocks:
            self._slices[block.name] = slice(cursor, cursor + block.size)
            cursor += block.size
        self._size = cursor

    @property
    def blocks(self) -> tuple[ParameterBlock, ...]:
        return self._blocks

    @property
    def n_parameters(self) -> int:
        return self._size

    def slice(self, name: str) -> slice:
        try:
            return self._slices[name]
        except KeyError as error:
            raise KeyError(f"Unknown parameter block {name!r}.") from error

    def bounds(self) -> VariableBounds:
        return VariableBounds(
            lower=np.concatenate([block.bounds().lower for block in self._blocks]),
            upper=np.concatenate([block.bounds().upper for block in self._blocks]),
        )

    def unpack(self, parameters: ArrayLike) -> dict[str, FloatArray]:
        vector = np.asarray(parameters, dtype=float)
        if vector.shape != (self.n_parameters,):
            raise ValueError(
                f"Expected a vector of size {self.n_parameters}; got {vector.shape}."
            )
        return {
            name: vector[block_slice].copy()
            for name, block_slice in self._slices.items()
        }

    def pack(self, values: Mapping[str, ArrayLike]) -> FloatArray:
        missing = set(self._slices) - set(values)
        extra = set(values) - set(self._slices)
        if missing or extra:
            raise ValueError(f"Mismatched blocks. Missing={missing}, extra={extra}.")
        vector = np.empty(self.n_parameters, dtype=float)
        for block in self._blocks:
            value = np.asarray(values[block.name], dtype=float).reshape(-1)
            if value.size != block.size:
                raise ValueError(
                    f"Block {block.name!r} expects {block.size} values; got {value.size}."
                )
            vector[self.slice(block.name)] = value
        return vector
