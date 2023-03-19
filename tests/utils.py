from typing import Any

from verboseindex.indexing import _parse_indexing_part, _parse_space_separated_dimensions, arange_at_position

Array = Any


def arange_at(xp, n_axes, axis, axis_len):
    return arange_at_position(xp, n_axes, axis, axis_len, device=None)


def pseudo_random_tensor(xp, shape: list[int]):
    total_size = 1
    for x in shape:
        total_size *= x
    # xor and two prime numbers
    values = (xp.arange(total_size, dtype=xp.int64) * 613 ^ 1234567) % 1009
    return xp.reshape(values, shape)


def enumerate_indexer(xp, indexer_pattern: str, indexer: Array, sizes: dict[str, int]) -> dict[str, Array]:
    """returns a dictionary with 1-dim arrays"""
    index_axes, index_other_axes = _parse_indexing_part(indexer_pattern)
    expected_shape = [indexer.shape[0]] + [sizes[axis] for axis in index_other_axes]
    assert indexer.shape == tuple(expected_shape)
    _template = indexer[0, ...] * 0
    result = {}
    for i, axis in enumerate(index_axes):
        result[axis] = xp.reshape(indexer[i, ...], (-1,))
    for j, axis in enumerate(index_other_axes):
        result[axis] = xp.reshape(
            _template + arange_at(xp, n_axes=len(index_other_axes), axis=j, axis_len=sizes[axis]), (-1,)
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
