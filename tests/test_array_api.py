from typing import TypeVar

from eindex._core import CompositionDecomposition
from eindex.array_api import _ArrayApiIXP, argmax, argmin, argsort, _einindex, gather

# gather_scatter,
# scatter,

from .utils import (
    _enum_1d,
    compose_index,
    enumerate_indexer,
    flatten,
    generate_array,
    generate_indexer,
    pseudo_random_tensor,
    range_of_shape,
    to_flat_index,
)

T = TypeVar("T")


# TODO tests for inf, +inf, -inf, nan?


def test_composition_and_decomposition():
    import numpy.array_api as xp

    ixp = _ArrayApiIXP(xp)

    x = range_of_shape(2, 3, 5, 7, xp=xp)
    comp = CompositionDecomposition(
        decomposed_shape=["a", "b", "c", "d"],
        composed_shape=[["a", "b"], ["c", "d"]],
    )
    x_composed = comp.compose_ixp(ixp, x, known_axes_lengths={})
    assert x_composed.shape == (2 * 3, 5 * 7)
    assert xp.all(x_composed == xp.reshape(x, (2 * 3, 5 * 7)))

    y = CompositionDecomposition(
        decomposed_shape=["a", "b", "c", "d"],
        composed_shape=[["a", "b"], [], ["c", "d"], []],
    ).compose_ixp(ixp, x, {})
    assert y.shape == (2 * 3, 1, 5 * 7, 1)
    assert xp.all(xp.reshape(x, (-1,)) == xp.reshape(y, (-1,)))

    comp = CompositionDecomposition(
        decomposed_shape=["a", "b", "e", "c", "d"],
        composed_shape=[["e", "c"], ["b"], ["a", "d"]],
    )
    x = range_of_shape(2, 3, 5, 7, 3, xp=xp)

    axes = {}
    y = comp.compose_ixp(ixp, x, axes)
    assert y.shape == (5 * 7, 3, 2 * 3)
    y_manual = xp.reshape(xp.permute_dims(x, (2, 3, 1, 0, 4)), y.shape)

    assert xp.all(y == y_manual)
    x2 = comp.decompose_ixp(ixp, y, axes)
    assert xp.all(x == x2)


def test_simple_indexing():
    import numpy.array_api as np

    # simple 2d test
    arr = pseudo_random_tensor(np, [5, 7])
    ind = np.arange(7) % 5
    x = _einindex("j <- i j, [i] j", arr, [ind])
    for j, i in _enum_1d(ind):
        assert arr[i, j] == x[j]

    y = _einindex("j <- j i, [i] j", np.permute_dims(arr, (1, 0)), [ind])
    for j, i in _enum_1d(ind):
        assert arr[i, j] == y[j]


def test_multidimensional_indexing():
    import numpy.array_api as xp

    ixp = _ArrayApiIXP(xp)

    B, H, W, C, T = 2, 3, 5, 7, 11
    hindices_bt = pseudo_random_tensor(xp, [B, T]) % H
    windices_bt = pseudo_random_tensor(xp, [B, T]) % W
    _t = hindices_bt

    embedding_bhwc = (
        0
        + ixp.arange_at_position(4, 0, B, _t) * 1000
        + ixp.arange_at_position(4, 1, H, _t) * 100
        + ixp.arange_at_position(4, 2, W, _t) * 10
        + ixp.arange_at_position(4, 3, C, _t) * 1
    )

    # imagine that you have pairs of image <> sentence.
    # goal is to get most suitable token from image for every token in sentence
    # thus for every token in sentence you compute best H and vW

    result = _einindex("c t b <- b H W c, [H, W] b t", embedding_bhwc, [hindices_bt, windices_bt])
    # example of using a single array for indexing multiple axes
    hw_indices_bt = xp.stack([hindices_bt, windices_bt])
    result2 = _einindex("c t b <- b H W c, [H, W] b t", embedding_bhwc, hw_indices_bt)
    assert xp.all(result == result2)

    # check vs manual element computation
    result_manual = result * 0
    for b in range(B):
        for t in range(T):
            for c in range(C):
                h = int(hindices_bt[b, t])
                w = int(windices_bt[b, t])
                result_manual[c, t, b] = embedding_bhwc[b, h, w, c]

    assert xp.all(result == result_manual)


def test_reverse_indexing():
    import numpy.array_api as xp

    ixp = _ArrayApiIXP(xp)

    C, T, B = 2, 3, 5
    # G = GPU, batch-like varaible
    G = 4
    H = 7
    W = 9

    t_indices_gbhw = xp.reshape(xp.arange(G * B * H * W), (G, B, H, W)) % T
    _t = t_indices_gbhw

    arr_gtbc = (
        0
        + ixp.arange_at_position(4, 0, G, _t) * 1000
        + ixp.arange_at_position(4, 1, T, _t) * 100
        + ixp.arange_at_position(4, 2, B, _t) * 10
        + ixp.arange_at_position(4, 3, C, _t) * 1
    )

    result = _einindex("g b c h w <- g t b c, [t] g b h w", arr_gtbc, [t_indices_gbhw])

    result_manual = result * 0
    for g in range(G):
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        t = int(t_indices_gbhw[g, b, h, w])
                        result_manual[g, b, c, h, w] = arr_gtbc[g, t, b, c]

    assert xp.all(result == result_manual)


def check_max_min(x, pattern: str):
    xp = x.__array_namespace__()
    assert xp.all(argmax(x, pattern) == argmin(-x, pattern))


def test_argmax_straight():
    import numpy.array_api as xp

    ixp = _ArrayApiIXP(xp)

    A, B, C, D = 2, 3, 5, 7
    x = pseudo_random_tensor(xp, [A, B, C, D])
    # set one maximum for every B, so argmax is unambiguous
    for b in range(B):
        x[1, b, b + 1, b + 2] = 2000 + b
    [a, b, c, d] = argmax(x, "a b c d -> [a, b, c, d]")
    assert x[a, b, c, d] == xp.max(x)
    cad = argmax(x, "a b c d -> [c, a, d] b")
    comp = CompositionDecomposition(composed_shape=[["c", "a", "d"], ["b"]], decomposed_shape=["a", "b", "c", "d"])
    reference = xp.argmax(comp.compose_ixp(ixp, x, {}), axis=0)
    assert xp.all(reference == compose_index(cad, [C, A, D]))


def test_argmax_by_indexing():
    import numpy.array_api as np

    x = np.reshape(np.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 10000
    reference = np.argmax(x, axis=0)

    assert np.all(argmax(x, "i j k -> [i] j k")[0, ...] == reference)
    assert np.all(argmax(x, "i j k -> [i] k j")[0, ...] == reference.T)

    ind = argmax(x, "i j k -> [i] j k")
    assert np.all(_einindex("j k <- i j k, [i] j k", x, ind) == np.max(x, axis=0))

    ind = argmax(x, "i j k -> [i, j] k")
    assert np.all(_einindex("k <- i j k, [i, j] k", x, ind) == np.max(x, axis=(0, 1)))

    ind = argmax(x, "i j k -> [j, i] k")
    assert np.all(_einindex("k <- i j k, [j, i] k", x, ind) == np.max(x, axis=(0, 1)))

    ind = argmax(x, "i j k -> [i, k] j")
    assert np.all(_einindex("j <- i j k, [i, k] j", x, ind) == np.max(x, axis=(0, 2)))

    ind = argmax(x, "i j k -> [k, i, j]")
    assert np.all(_einindex(" <- i j k, [k, i, j]", x, ind) == np.max(x))

    check_max_min(x, "i j k -> [k, i, j]")
    check_max_min(x, "i j k -> [i, j] k")
    check_max_min(x, "i j k -> [j, i] k")
    check_max_min(x, "i j k -> [j] k i")


def test_argsort_against_numpy():
    import numpy.array_api as np

    x = np.reshape(np.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 1000

    assert np.all(argsort(x, "i j k -> [i] order j k")[0, ...] == np.argsort(x, axis=0))
    right = np.permute_dims(np.argsort(x, axis=0), (2, 1, 0))
    assert np.all(argsort(x, "i j k -> [i] k j order")[0, ...] == right)

    ind = argsort(x, "i j k -> [k, i, j] order")
    assert np.all(_einindex("order <- i j k, [k, i, j] order", x, ind) == np.sort(np.reshape(x, (-1,))))

    ind = argsort(x, "i j k -> [k, i] order j")
    reference = np.permute_dims(x, (0, 2, 1))
    reference = np.reshape(reference, (-1, reference.shape[-1]))
    assert np.all(_einindex("order j <- i j k, [k, i] order j", x, ind) == np.sort(reference, axis=0))


def test_index():
    import numpy.array_api as xp

    ixp = _ArrayApiIXP(xp)

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

    array = generate_array(xp, "a b c d", sizes=sizes)
    indexer = generate_indexer(xp, "[a, c] d f g", sizes=sizes)
    result_einindex = _einindex("g f d b <- a b c d, [a, c] d f g", array, indexer)
    result = gather("g f d b <- a b c d, [a, c] d f g", array, indexer)
    print(result.shape, flatten(xp, result)[0])
    print(result_einindex.shape, flatten(xp, result_einindex)[0])
    indexer_as_dict = enumerate_indexer(ixp, "[a, c] d f g", indexer=indexer, sizes=sizes)

    for b in range(sizes["b"]):
        flat_index_arr = to_flat_index("a b c d", {**indexer_as_dict, "b": b}, sizes=sizes)
        flat_index_result = to_flat_index("g f d b", {**indexer_as_dict, "b": b}, sizes=sizes)

        array_flat = flatten(xp, array)
        result_flat = flatten(xp, result)

        for i, j in zip(flat_index_arr, flat_index_result, strict=True):
            assert array_flat[i] == result_flat[j], ("failed", i, j)

    assert xp.all(result_einindex == result), (result_einindex, result)


def test_gather():
    import numpy.array_api as np

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
    result_gather = gather(full_pattern, array, indexer, agg="sum")

    _ixp = _ArrayApiIXP(np)
    indexer_as_dict = enumerate_indexer(_ixp, index_pattern, indexer=indexer, sizes=sizes)

    array_flat = flatten(np, array)
    result_flat = flatten(np, result_gather)

    for d in range(sizes["d"]):
        flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)
        flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)

        for ia, ir in zip(flat_index_array, flat_index_final, strict=True):
            result_flat[ir] -= array_flat[ia]

    assert np.max(abs(result_flat)) == 0

    # checking different aggregations
    array_flat_float = np.astype(array_flat, np.float64)

    for agg_name, agg_func, default_value in [
        ("sum", lambda a, b: a + b, 0.0),
        ("min", min, np.inf),
        ("max", max, -np.inf),
    ]:
        result_gather = gather(full_pattern, array, indexer, agg=agg_name)
        result_gather = np.reshape(result_gather, (-1,))
        result_ref = np.full(shape=tuple(sizes[d] for d in final_pattern.split()), fill_value=default_value)
        result_ref = np.reshape(result_ref, (-1,))
        for d in range(sizes["d"]):
            flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)
            flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)

            for ia, ir in zip(flat_index_array, flat_index_final, strict=True):
                result_ref[ir] = agg_func(array_flat_float[ia], result_ref[ir])
        assert np.all(np.astype(result_ref, np.int64) == result_gather)

    # checking mean aggregation on constant tensor
    result_mean_const = gather(full_pattern, np.full(array.shape, fill_value=3.0), indexer, agg="mean")
    assert np.all(result_mean_const == 3.0)

    # testing that ratio is constant, as number of elements averaged is the same for every result entry
    values = np.astype(array**2, np.float64) + 1.0
    result_mean = gather(full_pattern, values, indexer, agg="mean")
    result__sum = gather(full_pattern, values, indexer, agg="sum")
    ratio = result_mean / result__sum
    assert np.min(ratio) * 0.99 < np.max(ratio) < np.min(ratio) * 1.01
