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
- support for multidimensional indexing.
  Indexing can use a list of indexers or a single tensor with multiple indexers.
- 'batch' indexing, when some axes of indexer and array should be matched
- extensible for (named) ellipses, including variadic number of indexers
- extensible for einops-style compositions and decompositions
- extensible for outer indexing when indexers are not aligned

"""


from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Tuple, TypeVar, Union

from . import VerboseIndexError
from ._parsing import ParsedPattern, _parse_indexing_part, _parse_space_separated_dimensions

T = TypeVar("T")
if TYPE_CHECKING:
    from numpy.array_api._array_object import Array

    T = TypeVar("T", bound=Array)

Aggregation = Literal["set", "min", "max", "sum", "mean", "std", "logsumexp"]

# which functions do we use?
# shape/ndim, reshape, transpose
# xp.zeros (create an empty index of correct shape, or create result of correct shape)
# 2d-indexing  over the first axis x_ij[i_k] -> y_kj
# 2d-reduction over the first axis x_ij[i_k] += y_kj -> result is x_ij # gather-scatter
# multi-reduction: x_ij[i_rk] += y_kj, which works like                # scatter
# the last two are not a part of array api standard


class CompositionDecomposition:
    """
    Minimal implementation of einops-style composition and decomposition of axes.
    Both 'compose' and 'decompose' memorize shapes of variables in the dictionary.
    """

    def __init__(
        self,
        decomposed_shape: List[str],
        composed_shape: List[List[str]],
    ):
        flat_shape = []
        for x in composed_shape:
            flat_shape.extend(x)

        self.compose_transposition: Tuple[int, ...] = tuple([decomposed_shape.index(x) for x in flat_shape])
        self.decompose_transposition: Tuple[int, ...] = tuple([flat_shape.index(x) for x in decomposed_shape])
        self.composed_shape = composed_shape
        self.decomposed_shape = decomposed_shape
        # If optimization engine of framework is good, we don't need these.
        # But we want to be sure we don't run unnecessary ops.
        self.needs_reshape = any(len(g) != 1 for g in self.composed_shape)
        self.needs_transposition = list(flat_shape) != list(self.composed_shape)

    def decompose_arapi(self, x, known_axes_lengths: dict[str, int]):
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
                        raise VerboseIndexError("Can't infer the size")

            if unknown_axis_name is None:
                assert shape[i] == known_sizes_prod
            else:
                known_axes_lengths[unknown_axis_name] = shape[i] // known_sizes_prod

            for axis in axis_group:
                flat_shape.append(known_axes_lengths[axis])

        if self.needs_reshape:
            x = xp.reshape(x, tuple(flat_shape))
        else:
            # will be removed
            assert x.shape == tuple(
                flat_shape
            ), f"shapes: {x.shape=} {flat_shape=} {self.composed_shape} {self.decomposed_shape}"
        if self.needs_transposition:
            return xp.permute_dims(x, self.decompose_transposition)
        else:
            return x

    def compose_arapi(self, x, known_axes_lengths: dict[str, int]):
        xp = x.__array_namespace__()

        for axis_len, axis_name in zip(x.shape, self.decomposed_shape, strict=True):
            if axis_name in known_axes_lengths:
                assert known_axes_lengths[axis_name] == axis_len
            else:
                known_axes_lengths[axis_name] = axis_len

        if self.needs_transposition:
            x = xp.permute_dims(x, self.compose_transposition)

        new_shape = []
        for axis_group in self.composed_shape:
            composed_axis_size = 1
            for axis_name in axis_group:
                composed_axis_size *= known_axes_lengths[axis_name]
            new_shape.append(composed_axis_size)

        if self.needs_reshape:
            x = xp.reshape(x, tuple(new_shape))
        else:
            # will be removed
            assert x.shape == tuple(new_shape)

        return x


def arange_at_position(xp, n_axes, axis, axis_len, device):
    x = xp.arange(axis_len, dtype=xp.int64, device=device)
    shape = [1] * n_axes
    shape[axis] = axis_len
    x = xp.reshape(x, shape)
    return x


def _prod(x: Iterable[int]) -> int:
    result = 1
    for el in x:
        result *= el
    return result


def _broadcast_shapes(shapes: List[Tuple[int, ...]]):
    # naive, but works
    return [max(axis_len_in_arrays) for axis_len_in_arrays in zip(*shapes, strict=True)]


def _index_to_list_array_api(ind) -> List:
    if isinstance(ind, list):
        return ind
    return [ind[i, ...] for i in range(ind.shape[0])]


# def _learn_axes_sizes(tensor, tensor_axes: list[str], known_axes_sizes: dict[str, int]):
#     for dim, axis_name in zip(tensor.shape, tensor_axes, strict=True):
#         if axis_name in known_axes_sizes:
#             assert dim == known_axes_sizes[axis_name]
#         else:
#             known_axes_sizes[axis_name] = dim


def compute_full_index(
    xp,
    ind: list,
    indexing_axes: list[str],
    indexer_other_axes_names: list[str],
    flat_index_over: list[str],
    known_axes_sizes: dict,
) -> Any:
    assert len(ind) == len(indexing_axes)
    device = ind[0].device
    for indexer in ind:
        assert len(indexer.shape) == len(indexer_other_axes_names)

    flat_index = 0
    shift = 1
    # NB: traversing in reverse direction
    # this implementation (compared to simpler one) is more 'parallelizable' as sum of integers is associative
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


class IndexFormula:
    def __init__(self, pattern: str):
        """
        :param pattern: example 'b t c <- b H W c, [H, W] b t'
        """
        self.pattern_parser = ParsedPattern(pattern=pattern)

        self.indexer_axes = []
        self.batch_axes = []
        self.result_and_index_axes = []
        self.result_and_array_axes = []

        for axis, presence in self.pattern_parser.axis2presence():
            if presence == (False, True, True, False):
                self.indexer_axes.append(axis)
            elif presence[2]:
                raise VerboseIndexError(f"Wrong usage of indexer variable {axis} in {pattern}")
            elif presence == (True, True, False, True):
                self.batch_axes.append(axis)
            elif presence == (True, False, False, True):
                self.result_and_index_axes.append(axis)
            elif presence == (True, True, False, False):
                self.result_and_array_axes.append(axis)
            else:
                raise VerboseIndexError(f"Axis '{axis}' is used incorrectly in {pattern}")

        # we will not use self.indexer_axes, because we need original order of axes
        assert set(self.indexer_axes) == set(self.pattern_parser.ind_axes_names)

        self.array_composition = CompositionDecomposition(
            decomposed_shape=self.pattern_parser.arr_axes_names,
            composed_shape=[
                self.batch_axes + self.pattern_parser.ind_axes_names,
                self.result_and_array_axes,
            ],
        )

        self.index_composition = CompositionDecomposition(
            decomposed_shape=self.pattern_parser.ind_other_axes_names,
            # single axis after composition
            composed_shape=[self.batch_axes + self.result_and_index_axes],
        )

        self.result_composition = CompositionDecomposition(
            decomposed_shape=self.pattern_parser.res_axes_names,
            composed_shape=[
                self.batch_axes + self.result_and_index_axes,
                self.result_and_array_axes,
            ],
        )

    def apply_to_array_api(self, arr: T, ind: Union[T, List[T]]):
        known_axes_sizes: dict[str, int] = {}
        xp = arr.__array_namespace__()
        ind_list = _index_to_list_array_api(ind)

        for indexer in ind_list:
            assert len(indexer.shape) == len(self.pattern_parser.ind_other_axes_names)

        # step 1. transpose, reshapes of arr; learn its dimensions
        arr_2d = self.array_composition.compose_arapi(arr, known_axes_sizes)

        # step 2. compute shifts and create an actual indexing array
        full_index = compute_full_index(
            xp,
            ind=ind_list,
            indexing_axes=self.pattern_parser.ind_axes_names,
            indexer_other_axes_names=self.pattern_parser.ind_other_axes_names,
            flat_index_over=self.batch_axes + self.pattern_parser.ind_axes_names,
            known_axes_sizes=known_axes_sizes,
        )

        # step 3. Flatten index
        full_index = self.index_composition.compose_arapi(full_index, known_axes_sizes)

        # step 4. indexing
        # python array api has xp.take, but it is not implemented anywhere
        # result_2d = xp.take(arr_2d, full_index, axis=0)
        result_2d = xp.stack([arr_2d[full_index[i], :] for i in range(full_index.shape[0])])

        # step 5. reshape result to correct form
        return self.result_composition.decompose_arapi(result_2d, known_axes_sizes)


class GatherFormula:
    def __init__(self, pattern: str, aggregation: Aggregation) -> None:
        """
        Example in which one aggregates the data
        'b t c <- b H W s c, [H, W] b t s replica'

        where multiple replicas are aggregated by plain sum
        """
        self.parsed_pattern = ParsedPattern(pattern)
        self.aggregation = aggregation

        self.indexer_axes = []  # H, W
        self.batch_axes = []  # b
        self.result_and_index_axes = []  # t
        self.result_and_array_axes = []  # c
        self.array_and_index_axes = []  # s
        self.index_only_axes = []  # replica

        for axis, presence in self.parsed_pattern.axis2presence():
            if presence == (False, True, True, False):
                self.indexer_axes.append(axis)
            elif presence == (True, True, False, True):
                self.batch_axes.append(axis)
            elif presence == (True, False, False, True):
                self.result_and_index_axes.append(axis)
            elif presence == (True, True, False, False):
                self.result_and_array_axes.append(axis)
            elif presence == (False, True, False, True):
                self.array_and_index_axes.append(axis)
            elif presence == (False, False, False, True):
                self.index_only_axes.append(axis)
            else:
                raise VerboseIndexError(f"Axis '{axis}' is used incorrectly in {pattern}")

        self.index_walks = self.batch_axes + self.indexer_axes + self.array_and_index_axes

        self.array_composition = CompositionDecomposition(
            decomposed_shape=self.parsed_pattern.arr_axes_names,
            composed_shape=[self.index_walks, self.result_and_array_axes],
        )

        self.index_composition = CompositionDecomposition(
            decomposed_shape=self.parsed_pattern.ind_other_axes_names,
            # [replicas, shared_between_index_and_result]
            composed_shape=[
                self.array_and_index_axes + self.index_only_axes,
                self.batch_axes + self.result_and_index_axes,
            ],
        )

        self.result_composition = CompositionDecomposition(
            decomposed_shape=self.parsed_pattern.res_axes_names,
            composed_shape=[self.batch_axes + self.result_and_index_axes, self.result_and_array_axes],
        )

    def apply_to_array_api(self, arr, ind):
        assert self.aggregation == "sum"
        known_axes_lengths: dict[str, int] = {}
        xp = arr.__array_namespace__()
        ind_list = _index_to_list_array_api(ind)

        for indexer in ind_list:
            assert len(indexer.shape) == len(self.parsed_pattern.ind_other_axes_names)

        # step 1. transpose, reshapes of arr; learn its dimensions
        arr_2d = self.array_composition.compose_arapi(arr, known_axes_lengths)

        # step 2. compute shifts and create an actual indexing array
        full_index_2d = compute_full_index(
            xp,
            ind=ind_list,
            indexing_axes=self.parsed_pattern.ind_axes_names,
            indexer_other_axes_names=self.parsed_pattern.ind_other_axes_names,
            flat_index_over=self.index_walks,
            known_axes_sizes=known_axes_lengths,
        )

        # step 3. Flatten index
        full_index_2d = self.index_composition.compose_arapi(full_index_2d, known_axes_lengths)

        # step 4. indexing
        cshape = self.result_composition.composed_shape
        shape = [_prod(known_axes_lengths[var] for var in group) for group in cshape]
        result_2d = xp.zeros(shape, dtype=arr.dtype, device=arr.device)

        for i in range(full_index_2d.shape[0]):
            for j in range(full_index_2d.shape[1]):
                result_2d[j, :] += arr_2d[full_index_2d[i, j], :]

        # step 5. reshape result to correct form
        return self.result_composition.decompose_arapi(result_2d, known_axes_lengths)


class ScatterFormula:
    def __init__(self, pattern: str, aggregation: Aggregation) -> None:
        """
        Performs scattering (aggregation in positions).
        Example of pattern: b s H W c <- b t c, [H W] b t s replica
        """
        self.aggregation = aggregation
        self.parsed_pattern = ParsedPattern(pattern=pattern)

        self.batch_axes = []  # b
        self.output_array_axes = []  # c
        self.output_index_axes = []  # H, W
        self.output_index_other = []  # s
        self.array_index_other = []  # t
        self.only_index_other_axes = []  # replica

        for axis, presence in self.parsed_pattern.axis2presence():
            if presence == (True, True, False, True):
                self.batch_axes.append(axis)
            elif presence == (True, True, False, False):
                self.output_array_axes.append(axis)
            elif presence == (True, False, True, False):
                self.output_index_axes.append(axis)
            elif presence == (True, False, False, True):
                self.output_index_other.append(axis)
            elif presence == (False, True, False, True):
                self.array_index_other.append(axis)
            elif presence == (False, False, False, True):
                self.only_index_other_axes.append(axis)
            else:
                raise VerboseIndexError(f"Axis {axis} is used incorrectly in '{pattern}'")

        self.index_walks = self.batch_axes + self.output_index_axes + self.output_index_other

        self.array_composition = CompositionDecomposition(
            decomposed_shape=self.parsed_pattern.arr_axes_names,
            composed_shape=[self.batch_axes + self.array_index_other, self.output_array_axes],
        )

        self.index_composition = CompositionDecomposition(
            decomposed_shape=self.parsed_pattern.ind_other_axes_names,
            # first axis is responsible for 'replicas',  when a single value can be scattered to multiple places
            composed_shape=[
                self.output_index_other + self.only_index_other_axes,
                self.batch_axes + self.array_index_other,
            ],
        )

        self.result_composition = CompositionDecomposition(
            decomposed_shape=self.parsed_pattern.res_axes_names,
            composed_shape=[self.index_walks, self.output_array_axes],
        )

    def apply_to_array_api(self, arr: T, ind: Union[T, List[T]], axis_sizes: dict[str, int]):
        assert self.aggregation == "sum"
        ind_list = _index_to_list_array_api(ind)
        xp = arr.__array_namespace__()
        known_axes_lengths = {**axis_sizes}

        # step 0. reshape arr to [(b t) (c)]
        arr_2d = self.array_composition.compose_arapi(arr, known_axes_lengths=known_axes_lengths)

        # step 1. build first index of shape [b t s replica] -> (b h w)
        # some output axes may be present only in ind_other_axes
        index_other_shape = _broadcast_shapes([x.shape for x in ind_list])
        for axis_len, axis in zip(index_other_shape, self.parsed_pattern.ind_other_axes_names, strict=True):
            if axis not in known_axes_lengths:
                known_axes_lengths[axis] = axis_len
            else:
                assert axis_len == known_axes_lengths[axis]

        first_flat_index = compute_full_index(
            xp,
            ind=ind_list,
            indexing_axes=self.parsed_pattern.ind_axes_names,
            indexer_other_axes_names=self.parsed_pattern.ind_other_axes_names,
            flat_index_over=self.index_walks,
            known_axes_sizes=known_axes_lengths,
        )

        # step 2. reshape flat index into [(s replica) (b t)] -> (b h w)
        flat_index_2d = self.index_composition.compose_arapi(first_flat_index, known_axes_lengths=known_axes_lengths)

        # step 3. creation of result in composed_shape
        cshape = self.result_composition.composed_shape
        shape = [_prod(known_axes_lengths[var] for var in group) for group in cshape]
        result = xp.zeros(shape, dtype=arr.dtype, device=arr.device)

        # step 4. aggregation
        # += does not work
        # result._array[flat_index_2d._array, ...] += arr_2d._array
        for i in range(flat_index_2d.shape[0]):
            for j in range(flat_index_2d.shape[1]):
                result[flat_index_2d[i, j], :] += arr_2d[j, :]

        return self.result_composition.decompose_arapi(result, known_axes_lengths=known_axes_lengths)


class GatherScatterFormula:
    def __init__(self, pattern: str, aggregation: Aggregation):
        """
        performs gather and scatter at the same time
        :param pattern: e.g 'b t H W2 c <- b H W c, [H, W, W2] b t order'
        """
        self.parsed_pattern = ParsedPattern(pattern)
        self.aggregation = aggregation

        self.batch_axes = []  # b
        self.input_indexer_axes = []  # h, w
        self.output_indexer_axes = []  # h, w2
        self.result_and_index_axes = []  # t
        self.result_and_array_axes = []  # c
        self.index_reduced_axes = []  # order

        for axis, presence in self.parsed_pattern.axis2presence():
            if presence == (False, True, True, False):
                self.input_indexer_axes.append(axis)
            elif presence == (True, False, True, False):
                self.output_indexer_axes.append(axis)
            elif presence == (True, True, True, False):
                # exception: I allow same indexer axis to be used in array and result
                self.input_indexer_axes.append(axis)
                self.output_indexer_axes.append(axis)
            elif presence == (True, True, False, True):
                self.batch_axes.append(axis)
            elif presence == (True, False, False, True):
                self.result_and_index_axes.append(axis)
            elif presence == (True, True, False, False):
                self.result_and_array_axes.append(axis)
            elif presence == (False, False, False, True):
                self.index_reduced_axes.append(axis)
            else:
                raise VerboseIndexError(f"Axis {axis} is used incorrectly in '{pattern}'")

        self.index1_walks = self.batch_axes + self.input_indexer_axes
        self.index2_walks = self.batch_axes + self.result_and_index_axes + self.output_indexer_axes
        # output = sum of output_indexer, batch, result_and_index, result_and_array
        # array = sum input_indexer, batch, result_and_array
        # indexing = union of input_indexer, output_indexer; they overlap
        # index_other = batch + result_and_index + reduced_index
        self.array_composition = CompositionDecomposition(
            decomposed_shape=self.parsed_pattern.arr_axes_names,
            composed_shape=[self.index1_walks, self.result_and_array_axes],
        )

        self.result_decomposition = CompositionDecomposition(
            decomposed_shape=self.parsed_pattern.res_axes_names,
            composed_shape=[self.index2_walks, self.result_and_array_axes],
        )

    def apply_to_array_api(self, arr: T, ind: Union[T, List[T]], axis_sizes: dict[str, int]):
        ind_list = _index_to_list_array_api(ind)
        xp = arr.__array_namespace__()
        known_axes_lengths = {**axis_sizes}

        # step 0. reshape arr to [(b h w) (c)]
        arr_2d = self.array_composition.compose_arapi(arr, known_axes_lengths=known_axes_lengths)

        # step 1. build first index of shape [b t order] -> (b h w)
        # some output axes may be present only in ind_other_axes
        index_other_shape = _broadcast_shapes([x.shape for x in ind_list])
        for axis_len, axis in zip(index_other_shape, self.parsed_pattern.ind_other_axes_names, strict=True):
            if axis not in known_axes_lengths:
                known_axes_lengths[axis] = axis_len
            else:
                assert axis_len == known_axes_lengths[axis]

        first_flat_index = compute_full_index(
            xp,
            ind=ind_list,
            indexing_axes=self.parsed_pattern.ind_axes_names,
            indexer_other_axes_names=self.parsed_pattern.ind_other_axes_names,
            flat_index_over=self.index1_walks,
            known_axes_sizes=known_axes_lengths,
        )
        # step 2. and take elements into [(b t order) (c)]
        # taken_2d = xp.take(arr_2d, xp.flatten(first_flat_index))
        taken_2d = xp.stack([arr_2d[i, ...] for i in xp.reshape(first_flat_index, (-1,))], axis=0)

        # step 3. build second index of shape [b t order] -> (b t h w2), put elements into [(b t h w2) (c)]
        flat_index_axes = self.index2_walks
        second_flat_index = compute_full_index(
            xp,
            ind=ind_list,
            indexing_axes=self.parsed_pattern.ind_axes_names,
            indexer_other_axes_names=self.parsed_pattern.ind_other_axes_names,
            flat_index_over=flat_index_axes,
            known_axes_sizes=known_axes_lengths,
        )
        # step 4.
        # convert index [b t order] -> (b t h w2) to [(b t order)] -> (b t h w2)
        # output would be [(b t h w2) (c)]
        first_axis = _prod(known_axes_lengths[axis] for axis in flat_index_axes)

        result_2d = xp.zeros([first_axis, taken_2d.shape[1]], dtype=arr.dtype)
        # doing indexing one-by-one
        for i, index in enumerate(xp.reshape(second_flat_index, (-1,))):
            result_2d[index, ...] += taken_2d[i, ...]

        # step 5
        return self.result_decomposition.decompose_arapi(result_2d, known_axes_lengths=known_axes_lengths)


class ArgFindFormula:
    def __init__(self, pattern: str, is_max: bool):
        """
        :param pattern: e.g 'b h w c -> [h, w] b c'
        """
        self.pattern = pattern
        left, right = pattern.split("->")
        self.input_axes = _parse_space_separated_dimensions(left)
        self.indexing_axes, self.indexing_other_axes = _parse_indexing_part(right)

        if len(self.indexing_axes) == 0:
            raise VerboseIndexError("At least one indexing axis should be in [...]")

        double_used_axes = set.intersection(set(self.indexing_axes), set(self.indexing_other_axes))
        if double_used_axes:
            raise VerboseIndexError(f"Some axes were used more than once: {double_used_axes}")

        on_one_side = set.symmetric_difference(
            set(self.input_axes), set.union(set(self.indexing_axes), set(self.indexing_other_axes))
        )
        if on_one_side:
            raise VerboseIndexError(
                f"All axes should be present in left and right side, but these are not: {on_one_side}"
            )

        self.transposition: list[int] = [self.input_axes.index(axis) for axis in self.indexing_other_axes]
        # note: we place indexing axes in reverse order here
        self.transposition += [self.input_axes.index(axis) for axis in self.indexing_axes[::-1]]
        self.is_max: bool = is_max

    def apply_to_array_api(self, arr):
        xp = arr.__array_namespace__()
        arr = xp.permute_dims(arr, self.transposition)

        reduced_shape = arr.shape[len(self.indexing_other_axes) :]
        result_shape = arr.shape[: len(self.indexing_other_axes)]
        arr = xp.reshape(arr, [*result_shape, -1])
        if self.is_max:
            flat_index = xp.argmax(arr, axis=-1)
        else:
            flat_index = xp.argmin(arr, axis=-1)
        result = []
        # shape here is also traversed in reverse order
        for axis_len in reduced_shape[::-1]:
            result.append(flat_index % axis_len)
            flat_index = flat_index // axis_len

        return xp.stack(result, axis=0)


class ArgmaxFormula(ArgFindFormula):
    def __init__(self, pattern: str) -> None:
        super().__init__(pattern=pattern, is_max=True)


class ArgminFormula(ArgFindFormula):
    def __init__(self, pattern: str) -> None:
        super().__init__(pattern=pattern, is_max=False)


class ArgsortFormula:
    def __init__(self, pattern: str, order_axis="order"):
        """
        :param pattern: e.g 'b h w c -> [h, w] b c order'
        """
        self.pattern = pattern
        left, right = pattern.split("->")
        self.input_axes = _parse_space_separated_dimensions(left)
        self.indexing_axes, self.indexing_other_axes = _parse_indexing_part(right)
        if order_axis in self.input_axes:
            raise VerboseIndexError(f"Special axis {order_axis} should not be in the left part of pattern: {pattern}")
        if order_axis not in self.indexing_other_axes:
            raise VerboseIndexError(
                f"Special axis {order_axis} should be in the result part outside of [...]: {pattern}"
            )

        repeated_axes = set.intersection(set(self.indexing_axes), set(self.indexing_other_axes))
        if repeated_axes:
            raise VerboseIndexError(f"Axes {repeated_axes} were repeated in result part: {pattern} ")

        difference = set.symmetric_difference(
            {*self.input_axes, order_axis},
            {*self.indexing_axes, *self.indexing_other_axes},
        )
        if difference:
            raise VerboseIndexError(f"Axes {difference} should be present both in input and result of {pattern}")

        self.transposition: list[int] = []
        self.position_of_order_axis = self.indexing_other_axes.index(order_axis)
        for axis in self.indexing_other_axes:
            if axis == order_axis:
                # note: we place indexing axes in reverse order here
                self.transposition += [self.input_axes.index(axis) for axis in self.indexing_axes[::-1]]
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


def argmax(tensor, pattern: str, /):
    formula = ArgmaxFormula(pattern)
    return formula.apply_to_array_api(tensor)


def argmin(tensor, pattern: str, /):
    formula = ArgminFormula(pattern)
    return formula.apply_to_array_api(tensor)


def argsort(tensor, pattern: str, /):
    formula = ArgsortFormula(pattern)
    return formula.apply_to_array_api(tensor)
