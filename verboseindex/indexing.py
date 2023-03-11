"""
Notation for verbose indexing-related operations.


Indexing tensors over multiple dimensions is quite counter-intuitive,
and there is no simple way to memorize this.

This notation aims to simplify this process.
Goal is to provide consistent set of operations related to indexing. 


Examples

1. Query a sequence of positions in an image.
   einindex('t c <- h w c, [h, w] t', arr_bhwc, [h_indices, w_indices])

   Same, but with a batch of images and a batch of sequences.
   Query for every token in sequence a token in the corresponding image.
   einindex('b t c <- b h w c, [h, w] b t', arr_bhwc, [h_indices_bt, w_indices_bt])

   This is equivalent, so you can pass indexers independently or together
   hw_indices_bt = np.asarray([h_indices_bt, w_indices_bt]
   einindex('b t c <- b h w c, [h, w] b t', arr_bhwc, hw_indices_bt))

   We always use first axis for indexing variables.
   For this reason [...] part should always go first in indexer.

   This makes the largest difference with Jonathan Malmaud's concept https://github.com/malmaud/einindex,
   which has almost identical grammar, but puts special dimension last, while we put it first.
   This trick allows naturally decomposing multiindex into individual dimensions or vice versa.


2. query for every token in the video the most suitable word in a (matching) sentence
   einindex('b t h w <- seq b, [seq] t b h w', arr_tbc, [seq_indices_tbhw])

   note, that only one indexer is used, but still it has to be enclosed in the list.
   That's a price for being generic. Alternatively leading singleton dimension can be added.


3. (not supported now, future planning)
   for every timeframe in a video, find the token with the highest norm (across h and w), and compose a new stack of them
   indices_2bt = argmax(x_bthwc.norm(dim=-1), 'b t h w -> [h, w] b t')
   selected_embeddings_btc = einindex('b t c <- b t h w c, [h, w] b t', x_bthwc, indices_2bt)

   while currently question is around 'how do we index',
   it is important to pre-align that with a question 'what are natural ways to get indices'.
   Most common are argmin/max. less common options: topk (works here), argsort, and random sampling.
   It is sufficient to just provide argsort. Topk and sampling can be build on the top of argsort



Some important properties of this notation:
- support for multiple indexers, including using a single tensor to keep multiple indexers
- 'batch' indexing, when some axes of indexer and array should be matched
- universal (one-indexing-to-rule-them-all)
- extensible for (named) ellipses, including variadic number of indexers
- extensible for einops-style compositions and decompositions
- extensible for outer indexing when indexers are not aligned

Current implementation based on python array api and uses loops,
because no appropriate indexing available in the standard.

"""

from typing import TYPE_CHECKING, List, Tuple, TypeVar, Union

from einops import EinopsError

T = TypeVar("T")
if TYPE_CHECKING:
    from numpy.array_api._array_object import Array
    T = TypeVar('T', bound=Array)


# which functions do we use?
# shape, reshape, transpose
# xp.zeros (create an empty index of correct shape)
# 2d-indexing  over the first axis x_ij[i_k] -> y_kj
# 2d-reduction over the first axis x_ij[i_k] += y_kj -> result is x_ij


class CompositionDecomposition:
    """
    Minimal implementation of einops-style composition and decomposition of axes.

    """
    def __init__(
        self,
        decomposed_shape: List[str],
        composed_shape: List[List[str]],
    ):
        flat_shape = []
        for x in composed_shape:
            flat_shape.extend(x)

        self.compose_transposition: Tuple[int, ...] = tuple(
            [decomposed_shape.index(x) for x in flat_shape]
        )
        self.decompose_transposition: Tuple[int, ...] = tuple(
            [flat_shape.index(x) for x in decomposed_shape]
        )
        self.composed_shape = composed_shape
        self.decomposed_shape = decomposed_shape

    def decompose(self, x, known_axes_lengths: dict[str, int]):
        xp = x.__array_namespace__()
        shape = x.shape

        flat_shape = []

        for i, axis_group in enumerate(self.composed_shape):
            unknown_axis_name = None
            known_sizes_prod = 1
            for axis_name in axis_group:
                if axis_name in known_axes_lengths:
                    known_sizes_prod *= known_axes_lengths[axis_name]
                else:
                    if unknown_axis_name is None:
                        unknown_axis_name = axis_name
                    else:
                        raise EinopsError("Can't infer the size")

            if unknown_axis_name is None:
                assert shape[i] == known_sizes_prod
            else:
                known_axes_lengths[unknown_axis_name] = shape[i] // known_sizes_prod

            for axis in axis_group:
                flat_shape.append(known_axes_lengths[axis])

        x = xp.reshape(x, flat_shape)
        return xp.permute_dims(x, self.decompose_transposition)

    def compose(self, x, known_axes_lengths: dict[str, int]):
        xp = x.__array_namespace__()

        for axis_len, axis_name in zip(x.shape, self.decomposed_shape):
            if axis_name in known_axes_lengths:
                assert known_axes_lengths[axis_name] == axis_len
            else:
                known_axes_lengths[axis_name] = axis_len

        x = xp.permute_dims(x, self.compose_transposition)
        new_shape = []
        for axis_group in self.composed_shape:
            composed_axis_size = 1
            for axis_name in axis_group:
                composed_axis_size *= known_axes_lengths[axis_name]
            new_shape.append(composed_axis_size)

        return xp.reshape(x, tuple(new_shape))


def arange_at_position(xp, n_axes, axis, axis_len, device=None):
    x = xp.arange(axis_len, dtype=xp.int64, device=device)
    shape = [1] * n_axes
    shape[axis] = axis_len
    x = xp.reshape(x, shape)
    return x


def _parse_indexing(x: str) -> Tuple[List[str], List[str]]:
    x = x.strip()
    if not x.startswith("["):
        raise EinopsError(
            f"composition axis should go first in indexer, like [h w] i j k, not {x}"
        )
    composition_start = x.index("[")
    composition_end = x.index("]")
    composition = x[composition_start + 1 : composition_end]
    ind_other_axes = x[composition_end + 1 :]
    indexing_axes_names = [x.strip() for x in composition.split(",")]
    indexer_other_axes_names = ind_other_axes.split()
    return indexing_axes_names, indexer_other_axes_names


def compute_full_index(
    xp,
    ind: list,
    indexing_axes: list[str],
    indexer_other_axes_names: list[str],
    flat_index_over: list[str],
    known_axes_sizes: dict,
):
    assert len(ind) == len(indexing_axes)
    flat_index = 0
    shift = 1
    device = ind[0].device

    # NB: traversing in reverse direction
    for axis_name in flat_index_over[::-1]:
        if axis_name in indexing_axes:
            indexer = ind[indexing_axes.index(axis_name)]
            flat_index = flat_index + shift * (indexer % known_axes_sizes[axis_name])
            shift *= known_axes_sizes[axis_name]
        else:
            axis_id = indexer_other_axes_names.index(axis_name)
            flat_index = (
                flat_index
                + arange_at_position(
                    xp,
                    len(indexer_other_axes_names),
                    axis=axis_id,
                    axis_len=known_axes_sizes[axis_name],
                    device=device,
                )
                * shift
            )
            shift *= known_axes_sizes[axis_name]

    return flat_index


class GatherFormula:
    def __init__(self, pattern: str):
        """
        :param pattern: example 'b t c <- b H W c, [H, W] b t'
        """
        self.pattern = pattern
        left, right = pattern.split("<-")
        arg_split = right.index(",")
        arr_pattern, ind_pattern = right[:arg_split], right[arg_split + 1 :]

        self.result_axes_names = left.split()
        self.array_axes_names = arr_pattern.split()
        self.indexing_axes_names, self.indexer_other_axes_names = _parse_indexing(
            ind_pattern
        )

        for group_name, group in [
            ("result", self.result_axes_names),
            ("array", self.array_axes_names),
            ("indexer", self.indexing_axes_names + self.indexer_other_axes_names),
        ]:
            if len(set(group)) != len(group):
                # need more verbosity, which axis
                # potentially we can allow duplicated axis in array if those are indexed variables
                raise EinopsError(
                    f"{group_name} pattern ({group}) contains a duplicated axis"
                )

        axis_groups = [
            self.result_axes_names,
            self.array_axes_names,
            self.indexing_axes_names,
            self.indexer_other_axes_names,
        ]

        all_axes = set()
        for group in axis_groups:
            all_axes.update(group)

        self.indexer_axes = []
        self.batch_axes = []
        self.result_and_index_axes = []
        self.result_and_array_axes = []

        for axis in all_axes:
            presence = tuple(axis in g for g in axis_groups)
            # want match-case here. sweet dreams
            if presence == (False, True, True, False):
                self.indexer_axes.append(axis)
            elif presence[2]:
                raise EinopsError(f"Wrong usage of indexer variable {axis}")
            elif presence == (True, True, False, True):
                self.batch_axes.append(axis)
            elif presence == (True, False, False, True):
                self.result_and_index_axes.append(axis)
            elif presence == (True, True, False, False):
                self.result_and_array_axes.append(axis)
            else:
                # TODO better categorization of wrong usage patterns
                raise EinopsError(f"{axis} is used incorrectly in {pattern}")

        assert set(self.indexer_axes) == set(self.indexing_axes_names)
        # order of these variables matters, since we can't lose mapping here
        self.indexer_axes = self.indexing_axes_names

        self.array_composition = CompositionDecomposition(
            decomposed_shape=self.array_axes_names,
            composed_shape=[
                self.batch_axes + self.indexer_axes,
                self.result_and_array_axes,
            ],
        )

        self.index_composition = CompositionDecomposition(
            decomposed_shape=self.indexer_other_axes_names,
            # single axis after composition
            composed_shape=[self.batch_axes + self.result_and_index_axes],
        )

        self.result_composition = CompositionDecomposition(
            decomposed_shape=self.result_axes_names,
            composed_shape=[
                self.batch_axes + self.result_and_index_axes,
                self.result_and_array_axes,
            ],
        )

    def apply_to_array_api(self, arr: T, ind: Union[T, List[T]]):
        known_axes_sizes: dict[str, int] = {}
        xp = arr.__array_namespace__()

        if not isinstance(ind, list):
            ind = [ind[i, ...] for i in range(ind.shape[0])]

        for indexer in ind:
            assert len(indexer.shape) == len(self.indexer_other_axes_names)

        # step 1. transpose, reshapes of arr; learn its dimensions
        arr_2d = self.array_composition.compose(arr, known_axes_sizes)

        # step 2. compute shifts and create an actual indexing array
        shift = 1
        full_index = xp.zeros(
            [1] * len(ind[0].shape), dtype=xp.int64, device=arr.device
        )

        # original order: [*batch-like axes, *indexing_axes,]
        # now we need to traverse them in the opposite direction

        for axis_name, indexer in list(zip(self.indexing_axes_names, ind))[::-1]:
            full_index = full_index + shift * (indexer % known_axes_sizes[axis_name])
            shift *= known_axes_sizes[axis_name]

        for axis_name in self.batch_axes[::-1]:
            axis_id = self.indexer_other_axes_names.index(axis_name)
            full_index = (
                full_index
                + arange_at_position(
                    xp,
                    len(self.indexer_other_axes_names),
                    axis=axis_id,
                    axis_len=known_axes_sizes[axis_name],
                    device=arr.device,
                )
                * shift
            )
            shift *= known_axes_sizes[axis_name]

        assert shift == arr_2d.shape[0]

        # step 3. Flatten index
        full_index = self.index_composition.compose(full_index, known_axes_sizes)

        # step 4. indexing
        # python array api has xp.take, but it is not implemented anywhere
        # did you know that there is conceptual programming ... just like art?
        # result_2d = xp.take(arr_2d, full_index, axis=0)
        result_2d = xp.stack(
            [arr_2d[full_index[i], :] for i in range(full_index.shape[0])]
        )

        # step 5. doing resulting
        result = self.result_composition.decompose(result_2d, known_axes_sizes)
        return result


class GatherScatterFormula:
    def __init__(self, pattern: str):
        """
        performs gather and scatter at the same time
        :param pattern: e.g 'b t h w2 c <- b h w c, [h, w, w2] b t order'
        """
        self.pattern = pattern
        output, input = pattern.split("<-")
        first_comma = input.index(",")
        array_pattern, index_pattern = input[:first_comma], input[first_comma + 1 :]
        self.result_axes = output.strip().split()
        self.array_axes = array_pattern.strip().split()
        self.index_axes, self.index_other_axes = _parse_indexing(index_pattern)

        groups = [
            self.result_axes,
            self.array_axes,
            self.index_axes,
            self.index_other_axes,
        ]

        all_axes = set()
        for g in groups:
            all_axes.update(g)

        self.batch_axes = []  # b
        self.input_indexer_axes = []  # h, w
        self.output_indexer_axes = []  # h, w2
        self.result_and_index_axes = []  # t
        self.result_and_array_axes = []  # c
        self.index_reduced_axes = []  # order

        for axis in all_axes:
            presence = tuple(axis in g for g in groups)
            if presence == (False, True, True, False):
                self.input_indexer_axes.append(axis)
            elif presence == (True, False, True, False):
                self.output_indexer_axes.append(axis)
            elif presence == (True, True, True, False):
                self.input_indexer_axes.append(axis)
                self.output_indexer_axes.append(axis)
            elif presence[2]:
                raise EinopsError(f"Wrong usage of indexer variable {axis}")
            elif presence == (True, True, False, True):
                self.batch_axes.append(axis)
            elif presence == (True, False, False, True):
                self.result_and_index_axes.append(axis)
            elif presence == (True, True, False, False):
                self.result_and_array_axes.append(axis)
            elif presence == (False, False, False, True):
                self.index_reduced_axes.append(axis)
            else:
                # TODO better categorization of wrong usage patterns
                raise EinopsError(f"{axis} is used incorrectly in {pattern}")

        # output = sum of output_indexer, batch, result_and_index, result_and_array
        # array = sum input_indexer, batch, result_and_array
        # indexing = union of input_indexer, output_indexer; they overlap
        # index_other = batch + result_and_index + reduced_index
        self.array_composition = CompositionDecomposition(
            decomposed_shape=self.array_axes,
            composed_shape=[
                self.batch_axes + self.input_indexer_axes,
                self.result_and_array_axes,
            ],
        )

        self.result_decomposition = CompositionDecomposition(
            decomposed_shape=self.result_axes,
            composed_shape=[
                self.batch_axes + self.result_and_index_axes + self.output_indexer_axes,
                self.result_and_array_axes,
            ],
        )

    def apply_to_array_api(
        self, arr: T, ind: Union[T, List[T]], axis_sizes: dict[str, int]
    ):
        if not isinstance(ind, list):
            ind = [ind[i, ...] for i in range(ind.shape[0])]
        xp = arr.__array_namespace__()
        known_axes_lengths = {**axis_sizes}

        # step 0. reshape arr to [(b h w) (c)]
        arr_2d = self.array_composition.compose(
            arr, known_axes_lengths=known_axes_lengths
        )

        # step 1. build first index of shape [b t order] -> (b h w)
        first_flat_index = compute_full_index(
            xp,
            ind=ind,
            indexing_axes=self.index_axes,
            indexer_other_axes_names=self.index_other_axes,
            flat_index_over=self.batch_axes + self.input_indexer_axes,
            known_axes_sizes=known_axes_lengths,
        )
        # step 2. and take elements into [(b t order) (c)]
        # taken_2d = xp.take(arr_2d, xp.flatten(first_flat_index))
        taken_2d = xp.stack(
            [arr_2d[i, ...] for i in xp.reshape(first_flat_index, (-1,))], axis=0
        )

        # step 3. build second index of shape [b t order] -> (b t h w2), put elements into [(b t h w2) (c)]
        flat_index_axes = (
            self.batch_axes + self.result_and_index_axes + self.output_indexer_axes
        )
        second_flat_index = compute_full_index(
            xp,
            ind=ind,
            indexing_axes=self.index_axes,
            indexer_other_axes_names=self.index_other_axes,
            flat_index_over=flat_index_axes,
            known_axes_sizes=known_axes_lengths,
        )
        # step 4.
        # convert index [b t order] -> (b t h w2) to [(b t order)] -> (b t h w2)
        # output would be [(b t h w2) (c)]
        first_axis = 1
        for axis in flat_index_axes:
            first_axis *= known_axes_lengths[axis]

        result_2d = xp.zeros([first_axis, taken_2d.shape[1]])
        # this is not a correct aggregation, but shows the idea
        for i, index in enumerate(xp.reshape(second_flat_index, (-1,))):
            result_2d[index, ...] = taken_2d[i, ...]

        # step 5
        return self.result_decomposition.decompose(
            result_2d, known_axes_lengths=known_axes_lengths
        )


class ArgFindFormula:
    def __init__(self, pattern: str):
        """
        :param pattern: e.g 'b h w c -> [h, w] b c'
        """
        self.pattern = pattern
        left, right = pattern.split("->")
        self.input_axes = left.strip().split()
        self.indexing_axes, self.indexing_other_axes = _parse_indexing(right)

        assert not set.intersection(
            set(self.indexing_axes), set(self.indexing_other_axes)
        )
        assert set(self.input_axes) == set.union(
            set(self.indexing_axes), set(self.indexing_other_axes)
        )

        self.transposition = [
            self.input_axes.index(axis) for axis in self.indexing_other_axes
        ]
        # note: we place indexing axes in reverse order here
        self.transposition += [
            self.input_axes.index(axis) for axis in self.indexing_axes[::-1]
        ]

    def apply_to_array_api(self, arr, is_max: bool):
        xp = arr.__array_namespace__()
        arr = xp.permute_dims(arr, self.transposition)

        reduced_shape = arr.shape[len(self.indexing_other_axes) :]
        result_shape = arr.shape[: len(self.indexing_other_axes)]
        arr = xp.reshape(arr, [*result_shape, -1])
        if is_max:
            flat_index = xp.argmax(arr, axis=-1)
        else:
            flat_index = xp.argmin(arr, axis=-1)
        result = []
        # shape here is also traversed in reverse order
        for axis_len in reduced_shape[::-1]:
            result.append(flat_index % axis_len)
            flat_index = flat_index // axis_len

        return xp.stack(result, axis=0)


class ArgsortFormula:
    def __init__(self, pattern: str, order_axis="order"):
        """
        :param pattern: e.g 'b h w c -> [h, w] b c order'
        """
        self.pattern = pattern
        left, right = pattern.split("->")
        self.input_axes = left.strip().split()
        self.indexing_axes, self.indexing_other_axes = _parse_indexing(right)
        assert order_axis in self.indexing_other_axes
        assert order_axis not in self.input_axes

        assert not set.intersection(
            set(self.indexing_axes), set(self.indexing_other_axes)
        )
        assert set(self.input_axes).union({order_axis}) == {
            *self.indexing_axes,
            *self.indexing_other_axes,
        }

        self.transposition: list[int] = []
        self.position_of_order_axis = self.indexing_other_axes.index(order_axis)
        for axis in self.indexing_other_axes:
            if axis == order_axis:
                # note: we place indexing axes in reverse order here
                self.transposition += [
                    self.input_axes.index(axis) for axis in self.indexing_axes[::-1]
                ]
            else:
                self.transposition.append(self.input_axes.index(axis))

    def apply_to_array_api(self, arr):
        xp = arr.__array_namespace__()
        arr = xp.permute_dims(arr, self.transposition)
        _l = self.position_of_order_axis
        _r = self.position_of_order_axis + len(self.indexing_axes)
        reduced_shape = arr.shape[_l:_r]

        arr = xp.reshape(arr, (*arr.shape[:_l], -1, *arr.shape[_r:]))
        flat_index = xp.argsort(arr, axis=_l)
        result = []
        # shape here is also traversed in reverse order
        for axis_len in reduced_shape[::-1]:
            result.append(flat_index % axis_len)
            flat_index = flat_index // axis_len

        return xp.stack(result, axis=0)


def einindex(pattern: str, arr: T, ind: Union[T, List[T]], /):
    formula = GatherFormula(pattern)
    return formula.apply_to_array_api(arr, ind)


def gather_scatter(pattern: str, arr: T, ind: Union[T, List[T]], /, **axis_sizes: int):
    formula = GatherScatterFormula(pattern=pattern)
    return formula.apply_to_array_api(arr, ind, axis_sizes=axis_sizes)


def argmax(tensor, pattern: str, /):
    formula = ArgFindFormula(pattern)
    return formula.apply_to_array_api(tensor, is_max=True)


def argmin(tensor, pattern: str, /):
    formula = ArgFindFormula(pattern)
    return formula.apply_to_array_api(tensor, is_max=False)


def argsort(tensor, pattern: str, /):
    formula = ArgsortFormula(pattern)
    return formula.apply_to_array_api(tensor)

