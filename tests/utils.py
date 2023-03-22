from typing import Any

from verboseindex._core import IXP, _index_to_list_array_api, _parse_indexing_part, _parse_space_separated_dimensions

Array = Any


def pseudo_random_tensor(xp, shape: list[int]):
    total_size = 1
    for x in shape:
        total_size *= x
    # xor and two prime numbers
    values = (xp.arange(total_size, dtype=xp.int64) * 613 ^ 1234567) % 1009
    return xp.reshape(values, shape)


def enumerate_indexer(ixp: IXP, indexer_pattern: str, indexer: Array, sizes: dict[str, int]) -> dict[str, Array]:
    """returns a dictionary with 1-dim arrays"""
    index_axes, index_other_axes = _parse_indexing_part(indexer_pattern)
    expected_shape = [indexer.shape[0]] + [sizes[axis] for axis in index_other_axes]
    assert indexer.shape == tuple(expected_shape)
    _template = indexer[0, ...] * 0
    result = {}
    for i, axis in enumerate(index_axes):
        result[axis] = ixp.xp.reshape(indexer[i, ...], (-1,))
    for j, axis in enumerate(index_other_axes):
        result[axis] = ixp.xp.reshape(
            _template
            + ixp.arange_at_position(
                n_axes=len(index_other_axes), axis=j, axis_len=sizes[axis], array_to_copy_device_from=_template
            ),
            (-1,),
        )

    return result


def to_flat_index(array_pattern: str, coordinates: dict[str, Array], sizes: dict[str, int]) -> Array:
    array_dims = _parse_space_separated_dimensions(array_pattern)
    result = 0
    for label in array_dims:
        result = result * sizes[label] + coordinates[label]

    return result


def generate_indexer(xp, indexer_pattern: str, sizes: dict[str, int]):
    index_axes, index_other_axes = _parse_indexing_part(indexer_pattern)

    shape = [len(index_axes)] + [sizes[axis] for axis in index_other_axes]
    indexer = pseudo_random_tensor(xp, shape=shape)
    for i, axis in enumerate(index_axes):
        indexer[i, ...] = indexer[i, ...] % sizes[axis]

    return indexer


def generate_array(xp, array_pattern: str, sizes: dict[str, int]):
    axes = _parse_space_separated_dimensions(array_pattern)
    shape = [sizes[axis] for axis in axes]

    return pseudo_random_tensor(xp=xp, shape=shape)


def range_of_shape(*shape: int, xp):
    res = 1
    for d in shape:
        res *= d
    return xp.reshape(xp.arange(res), shape)


def flatten(xp, x):
    return xp.reshape(x, (-1,))


def compose_index(indexers, shapes: list):
    indexers = _index_to_list_array_api(indexers)

    result = 0
    for indexer, axis_len in zip(indexers, shapes, strict=True):
        result = result * axis_len + indexer
    return result


def _enum_1d(arr):
    for i in range(arr.shape[0]):
        yield i, int(arr[i, ...])
