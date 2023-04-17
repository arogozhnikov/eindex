from typing import Any, List, Tuple, TypeVar

import numpy as np

from eindex._core import CompositionDecomposition
from eindex.numpy import _numpy_ixp, argmax, argmin, argsort, einindex, gather, gather_scatter, scatter

from .utils import (
    _enum_1d,
    compose_index,
    enumerate_indexer,
    flatten,
    generate_array,
    generate_indexer,
    range_of_shape,
    to_flat_index,
)

T = TypeVar("T")


def arange_at_position(xp, n_axes, axis, axis_len):
    x = xp.arange(axis_len, dtype=xp.int64)
    shape = [1] * n_axes
    shape[axis] = axis_len
    x = xp.reshape(x, shape)
    return x


def arange_at(xp, n_axes, axis, axis_len):
    return arange_at_position(xp, n_axes, axis, axis_len)


def pseudo_random_tensor(_, shape):
    from numpy.random import randint

    return randint(size=shape, low=0, high=10000)


def test_composition_and_decomposition():
    x = range_of_shape(2, 3, 5, 7, xp=np)
    comp = CompositionDecomposition(
        decomposed_shape=["a", "b", "c", "d"],
        composed_shape=[["a", "b"], ["c", "d"]],
    )
    x_composed = comp.compose_ixp(_numpy_ixp, x, known_axes_lengths={})
    assert x_composed.shape == (2 * 3, 5 * 7)
    assert np.all(x_composed == np.reshape(x, (2 * 3, 5 * 7)))

    y = CompositionDecomposition(
        decomposed_shape=["a", "b", "c", "d"],
        composed_shape=[["a", "b"], [], ["c", "d"], []],
    ).compose_ixp(_numpy_ixp, x, {})
    assert y.shape == (2 * 3, 1, 5 * 7, 1)
    assert np.all(np.reshape(x, (-1,)) == np.reshape(y, (-1,)))

    comp = CompositionDecomposition(
        decomposed_shape=["a", "b", "e", "c", "d"],
        composed_shape=[["e", "c"], ["b"], ["a", "d"]],
    )
    x = range_of_shape(2, 3, 5, 7, 3, xp=np)

    axes = {}
    y = comp.compose_ixp(_numpy_ixp, x, axes)
    assert y.shape == (5 * 7, 3, 2 * 3)
    y_manual = np.reshape(np.transpose(x, (2, 3, 1, 0, 4)), y.shape)

    assert np.all(y == y_manual)
    x2 = comp.decompose_ixp(_numpy_ixp, y, axes)
    assert np.all(x == x2)


def test_simple_indexing():
    # simple 2d test
    arr = pseudo_random_tensor(np, [5, 7])

    ind = np.arange(7) % 5
    x = einindex("j <- i j, [i] j", arr, [ind])
    for j, i in _enum_1d(ind):
        assert arr[i, j] == x[j]

    y = einindex("j <- j i, [i] j", np.transpose(arr, (1, 0)), [ind])
    for j, i in _enum_1d(ind):
        assert arr[i, j] == y[j]


def test_multidimensional_indexing():
    B, H, W, C, T = 2, 3, 5, 7, 11

    embedding_bhwc = (
        0
        + arange_at(np, 4, 0, B) * 1000
        + arange_at(np, 4, 1, H) * 100
        + arange_at(np, 4, 2, W) * 10
        + arange_at(np, 4, 3, C) * 1
    )

    hindices_bt = pseudo_random_tensor(np, [B, T]) % H
    windices_bt = pseudo_random_tensor(np, [B, T]) % W

    # imagine that you have pairs of image <> sentence.
    # goal is to get most suitable token from image for every token in sentence
    # thus for every token in sentence you compute best H and vW

    result = einindex("c t b <- b H W c, [H, W] b t", embedding_bhwc, [hindices_bt, windices_bt])
    # example of using a single array for indexing multiple axes
    hw_indices_bt = np.stack([hindices_bt, windices_bt])
    result2 = einindex("c t b <- b H W c, [H, W] b t", embedding_bhwc, hw_indices_bt)
    assert np.all(result == result2)

    # check vs manual element computation
    result_manual = result * 0
    for b in range(B):
        for t in range(T):
            for c in range(C):
                h = int(hindices_bt[b, t])
                w = int(windices_bt[b, t])
                result_manual[c, t, b] = embedding_bhwc[b, h, w, c]

    assert np.all(result == result_manual)


def test_reverse_indexing():
    C, T, B = 2, 3, 5
    # G = GPU, batch-like varaible
    G = 4
    H = 7
    W = 9

    arr_gtbc = (
        0
        + arange_at(np, 4, 0, G) * 1000
        + arange_at(np, 4, 1, T) * 100
        + arange_at(np, 4, 2, B) * 10
        + arange_at(np, 4, 3, C) * 1
    )

    t_indices_gbhw = np.reshape(np.arange(G * B * H * W), (G, B, H, W)) % T

    result = einindex("g b c h w <- g t b c, [t] g b h w", arr_gtbc, [t_indices_gbhw])

    result_manual = result * 0
    for g in range(G):
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        t = int(t_indices_gbhw[g, b, h, w])
                        result_manual[g, b, c, h, w] = arr_gtbc[g, t, b, c]

    assert np.all(result == result_manual)


def check_max_min(x, pattern: str):
    assert np.allclose(argmax(x, pattern), argmin(-x, pattern))


def test_argmax_straight():
    A, B, C, D = 2, 3, 5, 7
    x = pseudo_random_tensor(np, [A, B, C, D])
    # set one maximum for every B, so argmax is unambiguous
    for b in range(B):
        x[1, b, b + 1, b + 2] = 2000 + b
    [a, b, c, d] = argmax(x, "a b c d -> [a, b, c, d]")
    assert x[a, b, c, d] == np.max(x)
    cad = argmax(x, "a b c d -> [c, a, d] b")
    comp = CompositionDecomposition(composed_shape=[["c", "a", "d"], ["b"]], decomposed_shape=["a", "b", "c", "d"])
    reference = np.argmax(comp.compose_ixp(_numpy_ixp, x, {}), axis=0)
    assert np.all(reference == compose_index(cad, [C, A, D]))


def test_argmax_by_indexing():
    x = np.reshape(np.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 10000
    reference = np.argmax(x, axis=0)

    assert np.all(argmax(x, "i j k -> [i] j k")[0, ...] == reference)
    assert np.all(argmax(x, "i j k -> [i] k j")[0, ...] == reference.T)

    ind = argmax(x, "i j k -> [i] j k")
    assert np.all(einindex("j k <- i j k, [i] j k", x, ind) == np.max(x, axis=0))

    ind = argmax(x, "i j k -> [i, j] k")
    assert np.all(einindex("k <- i j k, [i, j] k", x, ind) == np.max(x, axis=(0, 1)))

    ind = argmax(x, "i j k -> [j, i] k")
    assert np.all(einindex("k <- i j k, [j, i] k", x, ind) == np.max(x, axis=(0, 1)))

    ind = argmax(x, "i j k -> [i, k] j")
    assert np.all(einindex("j <- i j k, [i, k] j", x, ind) == np.max(x, axis=(0, 2)))

    ind = argmax(x, "i j k -> [k, i, j]")
    assert np.all(einindex(" <- i j k, [k, i, j]", x, ind) == np.max(x))

    check_max_min(x, "i j k -> [k, i, j]")
    check_max_min(x, "i j k -> [i, j] k")
    check_max_min(x, "i j k -> [j, i] k")
    check_max_min(x, "i j k -> [j] k i")


def test_argsort_against_numpy():
    x = np.reshape(np.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 1000

    assert np.all(argsort(x, "i j k -> [i] order j k")[0, ...] == np.argsort(x, axis=0))
    right = np.transpose(np.argsort(x, axis=0), (2, 1, 0))
    assert np.all(argsort(x, "i j k -> [i] k j order")[0, ...] == right)

    ind = argsort(x, "i j k -> [k, i, j] order")
    assert np.all(einindex("order <- i j k, [k, i, j] order", x, ind) == np.sort(np.reshape(x, (-1,))))

    ind = argsort(x, "i j k -> [k, i] order j")
    reference = np.transpose(x, (0, 2, 1))
    reference = np.reshape(reference, (-1, reference.shape[-1]))
    assert np.all(einindex("order j <- i j k, [k, i] order j", x, ind) == np.sort(reference, axis=0))


def test_gather_scatter_runs():
    x = np.reshape(np.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 1000
    indices = np.arange(5)
    groups = np.arange(5) // 2 * 0

    result = gather_scatter("i j <- i j g, [g, i] order", x, [groups, indices])
    print(result)


def test_index():
    sizes = dict(
        a=2,
        b=3,
        c=5,
        d=7,
        e=2,
        f=3,
        g=4,
        h=5,
    )

    array = generate_array(np, "a b c d", sizes=sizes)
    indexer = generate_indexer(np, "[a, c] d f g", sizes=sizes)
    result1 = einindex("g f d b <- a b c d, [a, c] d f g", array, indexer)
    result2 = gather("g f d b <- a b c d, [a, c] d f g", array, indexer)
    assert np.allclose(result1, result2)
    indexer_as_dict = enumerate_indexer(_numpy_ixp, "[a, c] d f g", indexer=indexer, sizes=sizes)

    for b in range(sizes["b"]):
        flat_index_arr = to_flat_index("a b c d", {**indexer_as_dict, "b": b}, sizes=sizes)
        flat_index_result = to_flat_index("g f d b", {**indexer_as_dict, "b": b}, sizes=sizes)

        array_flat = flatten(np, array)
        result_flat = flatten(np, result1)

        for i, j in zip(flat_index_arr, flat_index_result, strict=True):
            assert array_flat[i] == result_flat[j]


def list_aggname_aggfunc_default_value() -> List[Tuple[str, Any, float]]:
    return [
        ("sum", lambda a, b: a + b, 0.0),
        ("min", min, np.inf),
        ("max", max, -np.inf),
    ]


def test_gather():
    sizes = dict(
        a=2,
        b=3,
        c=5,
        d=7,
        i1=3,
        i2=5,
        r=3,
    )

    final_pattern = "b c d"
    array_pattern = "b i1 i2 d r"
    index_pattern = "[i1, i2] c b a r"
    full_pattern = f"{final_pattern} <- {array_pattern}, {index_pattern}"
    array = generate_array(np, array_pattern=array_pattern, sizes=sizes)
    indexer = generate_indexer(np, index_pattern, sizes=sizes)
    result_gather = gather(full_pattern, array, indexer, aggregation="sum")
    indexer_as_dict = enumerate_indexer(_numpy_ixp, index_pattern, indexer=indexer, sizes=sizes)

    array_flat = flatten(np, array)
    result_flat = flatten(np, result_gather)

    for d in range(sizes["d"]):
        flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)
        flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)

        for ia, ir in zip(flat_index_array, flat_index_final, strict=True):
            result_flat[ir] -= array_flat[ia]

    assert np.max(abs(result_flat)) == 0

    # checking different aggregations

    for agg_name, agg_func, default_value in [
        ("sum", lambda a, b: a + b, 0.0),
        ("min", min, np.inf),
        ("max", max, -np.inf),
    ]:
        result_gather = gather(full_pattern, array, indexer, aggregation=agg_name).reshape(-1)
        result_ref = np.full(shape=[sizes[d] for d in final_pattern.split()], fill_value=default_value).reshape(-1)
        for d in range(sizes["d"]):
            flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)
            flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)

            for ia, ir in zip(flat_index_array, flat_index_final, strict=True):
                result_ref[ir] = agg_func(array_flat[ia], result_ref[ir])
        assert np.allclose(result_ref, result_gather)

    # checking mean aggregation on constant tensor
    result_mean_const = gather(full_pattern, array * 0 + 3.0, indexer, aggregation="mean")
    assert np.allclose(result_mean_const, 3.0)

    # testing that ratio is constant, as number of elements averaged is the same for every result entry
    result_mean = gather(full_pattern, array.clip(0) + 1.0, indexer, aggregation="mean")
    result__sum = gather(full_pattern, array.clip(0) + 1.0, indexer, aggregation="sum")
    ratio = result_mean / result__sum
    assert ratio.min() * 0.99 < ratio.max() < ratio.min() * 1.01


def test_scatter():
    sizes = dict(
        b=3,
        c=5,
        d=7,
        e=2,
        f=3,
        g=4,
        h=5,
    )

    array_pattern = "b c d"
    index_pattern = "[f, h] c b e"
    final_pattern = "b f h d e"
    full_pattern: str = f"{final_pattern} <- {array_pattern}, {index_pattern}"
    array = generate_array(np, array_pattern=array_pattern, sizes=sizes)
    indexer = generate_indexer(np, index_pattern, sizes=sizes)
    result = scatter(full_pattern, array, indexer, f=sizes["f"], h=sizes["h"])
    indexer_as_dict = enumerate_indexer(_numpy_ixp, index_pattern, indexer=indexer, sizes=sizes)

    array_flat = flatten(np, array)
    result_flat = flatten(np, result)

    for d in range(sizes["d"]):
        flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)
        flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)

        for ia, ir in zip(flat_index_array, flat_index_final, strict=True):
            result_flat[ir] -= array_flat[ia]

    assert np.max(abs(result_flat)) == 0

    # check different aggregations
    for agg_name, agg_func, default_value in list_aggname_aggfunc_default_value():
        result_scatter = scatter(
            full_pattern, array, indexer, aggregation=agg_name, f=sizes["f"], h=sizes["h"]
        ).reshape(-1)
        result_ref = np.full(
            shape=[sizes[d] for d in final_pattern.split()], fill_value=default_value, dtype=array.dtype
        ).reshape(-1)
        for d in range(sizes["d"]):
            flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)
            flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)

            for ia, ir in zip(flat_index_array, flat_index_final, strict=True):
                result_ref[ir] = agg_func(array_flat[ia], result_ref[ir])
        assert np.allclose(result_ref, result_scatter)

    # testing aggregation on constant
    result = scatter(full_pattern, array * 0 + 3.0, indexer, aggregation="mean", f=sizes["f"], h=sizes["h"])
    result_sum = scatter(full_pattern, array * 0 + 1.0, indexer, aggregation="sum", f=sizes["f"], h=sizes["h"])
    assert np.allclose(result[result_sum > 0], 3.0)
    assert np.all(np.isnan(result[result_sum == 0]))


def test_gather_scatter():
    sizes = dict(
        b=3,
        c=5,
        r=3,
        f=4,
        i1=2,
        i2=3,
        i3=5,
    )

    final_pattern = "             b c i1 i3 f  "
    array_pattern = "             b c i1 i2    "
    index_pattern = "[i1, i2, i3] b         f r"
    full_pattern = f"{final_pattern} <- {array_pattern}, {index_pattern} "
    array = generate_array(np, array_pattern=array_pattern, sizes=sizes)
    indexer = generate_indexer(np, index_pattern, sizes=sizes)
    result = gather_scatter(full_pattern, array, indexer, i3=sizes["i3"])
    indexer_as_dict = enumerate_indexer(_numpy_ixp, index_pattern, indexer=indexer, sizes=sizes)

    array_flat = flatten(np, array)
    result_flat = flatten(np, result)

    for c in range(sizes["c"]):
        flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "c": c}, sizes=sizes)
        flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "c": c}, sizes=sizes)

        for ia, ir in zip(flat_index_array, flat_index_final, strict=True):
            result_flat[ir] -= array_flat[ia]

    assert np.max(abs(result_flat)) == 0

    # check different aggregations
    for agg_name, agg_func, default_value in list_aggname_aggfunc_default_value():
        result_gst = gather_scatter(full_pattern, array, indexer, aggregation=agg_name, i3=sizes["i3"]).reshape(-1)
        result_ref = np.full(
            shape=[sizes[d] for d in final_pattern.split()], fill_value=default_value, dtype=array.dtype
        ).reshape(-1)
        for c in range(sizes["c"]):
            flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "c": c}, sizes=sizes)
            flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "c": c}, sizes=sizes)

            for ia, ir in zip(flat_index_array, flat_index_final, strict=True):
                result_ref[ir] = agg_func(array_flat[ia], result_ref[ir])
        assert np.allclose(result_ref, result_gst)

    # check aggregation of constant
    result_mean = gather_scatter(full_pattern, array * 0 + 3.0, indexer, aggregation="mean", i3=sizes["i3"])
    result_sum = gather_scatter(full_pattern, array * 0 + 1, indexer, aggregation="sum", i3=sizes["i3"])
    assert np.all(result_mean[result_sum > 0] == 3)
    assert np.all(np.isnan(result_mean[result_sum == 0]))


# TODO test gatherscatter with combination of scatter and gather
# TODO test several patterns per combination?
# TODO get a way to deal with indexers
