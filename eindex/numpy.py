from typing import Any, List, Optional, Union

import numpy as np

from . import _core
from ._core import Aggregation

__all__ = ["argmax", "argmin", "argsort", "gather", "scatter", "gather_scatter"]

Array = np.ndarray


class _NumpyIXP(_core.IXP):
    xp: Any

    def __init__(self) -> None:
        self.xp = np

    def permute_dims(self, arr, permutation):
        return np.transpose(arr, permutation)

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from):
        # array_to_copy_device_from is ignored as numpy supports only CPU
        x = np.arange(axis_len, dtype=np.int64)
        shape = [1] * n_axes
        shape[axis] = axis_len
        return np.reshape(x, shape)


_numpy_ixp = _NumpyIXP()


def argmax(tensor: Array, pattern: str, /) -> Array:
    formula = _core.ArgmaxFormula(pattern)
    return formula.apply_to_ixp(_numpy_ixp, tensor)


def argmin(tensor: Array, pattern: str, /) -> Array:
    formula = _core.ArgminFormula(pattern)
    return formula.apply_to_ixp(_numpy_ixp, tensor)


def argsort(tensor: Array, pattern: str, /) -> Array:
    formula = _core.ArgsortFormula(pattern)
    return formula.apply_to_ixp(_numpy_ixp, tensor)


def _einindex(pattern: str, arr: Array, ind: Union[Array, List[Array]], /):
    formula = _core.IndexFormula(pattern)
    return formula.apply_to_numpy(_numpy_ixp, arr, ind)


def gather(pattern: str, arr: Array, ind: Union[Array, List[Array]], agg: Optional[Aggregation] = None):
    formula = _core.GatherFormula(pattern=pattern, agg=agg)
    return formula.apply_to_numpy(_numpy_ixp, arr, ind)


def gather_scatter(
    pattern: str, arr: Array, ind: Union[Array, List[Array]], /, agg: Aggregation = "sum", **axis_sizes: int
):
    formula = _core.GatherScatterFormula(pattern, agg=agg)
    return formula.apply_to_numpy(_numpy_ixp, arr, ind, axis_sizes=axis_sizes)


def scatter(
    pattern: str, arr: Array, ind: Union[Array, List[Array]], /, agg: Aggregation = "sum", **axis_sizes: int
):
    formula = _core.ScatterFormula(pattern, agg=agg)
    return formula.apply_to_numpy(_numpy_ixp, arr, ind, axis_sizes=axis_sizes)
