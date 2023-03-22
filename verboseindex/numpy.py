from typing import Any, List, Union

import numpy as np

from ._core import (
    IXP,
    Aggregation,
    ArgmaxFormula,
    ArgminFormula,
    ArgsortFormula,
    GatherFormula,
    GatherScatterFormula,
    IndexFormula,
    ScatterFormula,
)

Array = np.ndarray


class NumpyIXP(IXP):
    xp: Any

    def __init__(self) -> None:
        self.xp = np

    def permute_dims(self, arr, permutation):
        return np.transpose(arr, permutation)

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from):
        x = np.arange(axis_len, dtype=np.int64)
        shape = [1] * n_axes
        shape[axis] = axis_len
        return np.reshape(x, shape)


numpy_ixp = NumpyIXP()


def argmax(tensor: Array, pattern: str, /) -> Array:
    formula = ArgmaxFormula(pattern)
    return formula.apply_to_ixp(numpy_ixp, tensor)


def argmin(tensor: Array, pattern: str, /) -> Array:
    formula = ArgminFormula(pattern)
    return formula.apply_to_ixp(numpy_ixp, tensor)


def argsort(tensor: Array, pattern: str, /) -> Array:
    formula = ArgsortFormula(pattern)
    return formula.apply_to_ixp(numpy_ixp, tensor)


def einindex(pattern: str, arr: Array, ind: Union[Array, List[Array]], /):
    formula = IndexFormula(pattern)
    return formula.apply_to_numpy(numpy_ixp, arr, ind)


def gather(pattern: str, arr: Array, ind: Union[Array, List[Array]], aggregation: Aggregation = "sum"):
    formula = GatherFormula(pattern=pattern, aggregation=aggregation)
    return formula.apply_to_numpy(numpy_ixp, arr, ind)


def gather_scatter(
    pattern: str, arr: Array, ind: Union[Array, List[Array]], /, aggregation: Aggregation = "sum", **axis_sizes: int
):
    formula = GatherScatterFormula(pattern, aggregation=aggregation)
    return formula.apply_to_numpy(numpy_ixp, arr, ind, axis_sizes=axis_sizes)


def scatter(
    pattern: str, arr: Array, ind: Union[Array, List[Array]], /, aggregation: Aggregation = "sum", **axis_sizes: int
):
    formula = ScatterFormula(pattern, aggregation=aggregation)
    return formula.apply_to_numpy(numpy_ixp, arr, ind, axis_sizes=axis_sizes)
