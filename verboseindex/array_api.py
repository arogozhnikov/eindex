from typing import List, TypeVar, Union

from ._core import (
    Aggregation,
    ArgmaxFormula,
    ArgminFormula,
    ArgsortFormula,
    GatherFormula,
    GatherScatterFormula,
    IndexFormula,
    ScatterFormula,
)

T = TypeVar("T")


def argmax(tensor: T, pattern: str, /) -> T:
    formula = ArgmaxFormula(pattern)
    return formula.apply_to_array_api(tensor)


def argmin(tensor: T, pattern: str, /) -> T:
    formula = ArgminFormula(pattern)
    return formula.apply_to_array_api(tensor)


def argsort(tensor: T, pattern: str, /) -> T:
    formula = ArgsortFormula(pattern)
    return formula.apply_to_array_api(tensor)


def einindex(pattern: str, arr: T, ind: Union[T, List[T]], /):
    formula = IndexFormula(pattern)
    return formula.apply_to_array_api(arr, ind)


def gather(pattern: str, arr: T, ind: Union[T, List[T]], aggregation: Aggregation = "sum"):
    formula = GatherFormula(pattern=pattern, aggregation=aggregation)
    return formula.apply_to_array_api(arr, ind)


def gather_scatter(
    pattern: str, arr: T, ind: Union[T, List[T]], /, aggregation: Aggregation = "sum", **axis_sizes: int
):
    formula = GatherScatterFormula(pattern, aggregation=aggregation)
    return formula.apply_to_array_api(arr, ind, axis_sizes=axis_sizes)


def scatter(pattern: str, arr: T, ind: Union[T, List[T]], /, aggregation: Aggregation = "sum", **axis_sizes: int):
    formula = ScatterFormula(pattern, aggregation=aggregation)
    return formula.apply_to_array_api(arr, ind, axis_sizes=axis_sizes)
