from verboseindex.indexing import (
    CompositionDecomposition,
    einindex,
    gather_scatter, 
    argmax, 
    argmin, 
    argsort,
    arange_at_position,
)

def test_composition_and_decomposition():
    import numpy.array_api as np

    x = np.arange(2 * 3 * 5 * 7)
    x = np.reshape(x, (2, 3, 5, 7))
    comp = CompositionDecomposition(
        decomposed_shape=["a", "b", "c", "d"],
        composed_shape=[["a", "b"], ["c", "d"]],
    )
    assert comp.compose(x, known_axes_lengths={}).shape == (2 * 3, 5 * 7)

    y = CompositionDecomposition(
        decomposed_shape=["a", "b", "c", "d"],
        composed_shape=[["a", "b"], [], ["c", "d"]],
    ).compose(x, {})
    assert y.shape == (2 * 3, 1, 5 * 7)
    assert np.all(np.reshape(x, (-1,)) == np.reshape(y, (-1,)))

    comp = CompositionDecomposition(
        decomposed_shape=["a", "b", "e", "c", "d"],
        composed_shape=[["e", "c"], ["b"], ["a", "d"]],
    )
    x = np.arange(2 * 3 * 5 * 7 * 3)
    x = np.reshape(x, (2, 3, 5, 7, 3))

    axes = {}
    y = comp.compose(x, axes)
    x2 = comp.decompose(y, axes)
    assert np.all(x == x2)




def test_simple_indexing():
    import numpy.array_api as np

    # simple 2d test
    arr = np.reshape(np.arange(5 * 7), (5, 7))
    ind = np.arange(7) % 5
    x = einindex("j <- i j, [i] j", arr, [ind])
    for j, i in enumerate(ind):
        assert arr[i, j] == x[j]

    y = einindex("j <- j i, [i] j", np.permute_dims(arr, (1, 0)), [ind])
    for j, i in enumerate(ind):
        assert arr[i, j] == y[j]


def test_multidimensional_indexing():
    import numpy.array_api as np

    embedding_bhwc = (
        +arange_at_position(np, 4, 0, 2) * 1000
        + arange_at_position(np, 4, 1, 3) * 100
        + arange_at_position(np, 4, 2, 5) * 10
        + arange_at_position(np, 4, 3, 7) * 1
    )

    hindices_bt = np.reshape(np.arange(6), (2, 3)) % 3
    windices_bt = np.reshape(np.arange(6), (2, 3)) % 5

    # imagine that you have pairs of image <> sentence
    # your goal is to get most suitable token from image for every token in sentence
    # thus for every token in sentence you compute best k and v

    result = einindex(
        "c t b <- b h w c, [h, w] b t", embedding_bhwc, [hindices_bt, windices_bt]
    )
    # example of using a single array for indexing multiple axes
    hw_indices_bt = np.stack([hindices_bt, windices_bt])
    result2 = einindex("c t b <- b h w c, [h, w] b t", embedding_bhwc, hw_indices_bt)
    assert np.all(result == result2)

    # check vs manual element computation
    result_manual = result * 0
    for b in range(2):
        for t in range(3):
            for c in range(7):
                h = int(hindices_bt[b, t])
                w = int(windices_bt[b, t])
                result_manual[c, t, b] = embedding_bhwc[b, h, w, c]

    assert np.all(result == result_manual)


def test_reverse_indexing():
    import numpy.array_api as np

    C, T, B = 2, 3, 5
    # G = GPU, batch-like varaible
    G = 4
    H = 7
    W = 9

    arr_gtbc = (
        +arange_at_position(np, 4, 0, G) * 1000
        + arange_at_position(np, 4, 1, T) * 100
        + arange_at_position(np, 4, 2, B) * 10
        + arange_at_position(np, 4, 3, C) * 1
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


def test_einargmax():
    import numpy.array_api as np

    x = np.reshape(np.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 1000
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


def test_einargsort():
    import numpy.array_api as np

    x = np.reshape(np.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 1000

    assert np.all(argsort(x, "i j k -> [i] order j k")[0, ...] == np.argsort(x, axis=0))
    right = np.permute_dims(np.argsort(x, axis=0), (2, 1, 0))
    assert np.all(argsort(x, "i j k -> [i] k j order")[0, ...] == right)

    ind = argsort(x, "i j k -> [k, i, j] order")
    assert np.all(
        einindex("order <- i j k, [k, i, j] order", x, ind)
        == np.sort(np.reshape(x, (-1,)))
    )

    ind = argsort(x, "i j k -> [k, i] order j")
    reference = np.permute_dims(x, (0, 2, 1))
    reference = np.reshape(reference, (-1, reference.shape[-1]))
    assert np.all(
        einindex("order j <- i j k, [k, i] order j", x, ind)
        == np.sort(reference, axis=0)
    )


def test_gather_scatter():
    import numpy.array_api as np

    x = np.reshape(np.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 1000
    indices = np.arange(5)
    groups = np.arange(5) // 2 * 0

    result = gather_scatter("i j <- i j g, [g, i] order", x, [groups, indices])
    print(result)


test_gather_scatter()