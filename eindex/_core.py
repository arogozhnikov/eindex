"""
Core formulas for transformations
"""
from typing import Any, Iterable, List, Literal, Optional, Tuple, TypeVar, Union

from . import EindexError
from ._parsing import ParsedPattern, _parse_indexing_part, _parse_space_separated_dimensions

T = TypeVar("T")


Aggregation = Literal["min", "max", "sum", "mean"]  # "std", "logsumexp"

# which functions do we use?
# shape/ndim, reshape, transpose
# xp.zeros (create an empty index of correct shape, or create result of correct shape)
# 2d-indexing  over the first axis x_ij[i_k] -> y_kj                   # gather
# multi-reduction: x_ij[i_rk] += y_kj                                  # scatter
# the last two are not a part of array api standard


class IXP:
    xp: Any

    def permute_dims(self, arr, permutation):
        raise NotImplementedError()

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from):
        raise NotImplementedError()


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

    def decompose_ixp(self, ixp: IXP, x: T, known_axes_lengths: dict[str, int]) -> T:
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
                        raise EindexError("Can't infer the size")

            if unknown_axis_name is None:
                if not (shape[i] == known_sizes_prod):
                    raise EindexError(
                        f"Composed axis {axis_group} is expected to be {known_sizes_prod}, found: {shape[i]} "
                    )
            else:
                known_axes_lengths[unknown_axis_name] = shape[i] // known_sizes_prod

            for axis in axis_group:
                flat_shape.append(known_axes_lengths[axis])

        if self.needs_reshape:
            x = ixp.xp.reshape(x, tuple(flat_shape))
        if self.needs_transposition:
            x = ixp.permute_dims(x, self.decompose_transposition)
        return x

    def compose_ixp(self, ixp: IXP, x: T, known_axes_lengths: dict[str, int]) -> T:
        for axis_len, axis_name in zip(x.shape, self.decomposed_shape, strict=True):
            if axis_name in known_axes_lengths:
                if not (known_axes_lengths[axis_name] == axis_len):
                    raise EindexError(
                        f"Axis '{axis_name}' expected size: {known_axes_lengths[axis_name]}, found: {axis_len}"
                    )
            else:
                known_axes_lengths[axis_name] = axis_len

        if self.needs_transposition:
            x = ixp.permute_dims(x, self.compose_transposition)

        new_shape = []
        for axis_group in self.composed_shape:
            composed_axis_size = 1
            for axis_name in axis_group:
                composed_axis_size *= known_axes_lengths[axis_name]
            new_shape.append(composed_axis_size)

        if self.needs_reshape:
            x = ixp.xp.reshape(x, tuple(new_shape))
        else:
            # will be removed
            assert x.shape == tuple(new_shape)

        return x


def _prod(x: Iterable[int]) -> int:
    result = 1
    for el in x:
        result *= el
    return result


def _broadcast_shapes(shapes: List[Tuple[int, ...]]) -> List[int]:
    # naive, does not really verify shapes
    # number of dimensions should be the same
    return [max(axis_len_in_arrays) for axis_len_in_arrays in zip(*shapes, strict=True)]


def _index_to_list_array_api(ind) -> List:
    if isinstance(ind, list):
        return ind
    return [ind[i, ...] for i in range(ind.shape[0])]


def compute_full_index_ixp(
    ixp: IXP,
    ind: list,
    indexing_axes: list[str],
    indexer_other_axes_names: list[str],
    flat_index_over: list[str],
    known_axes_sizes: dict,
) -> Any:
    if len(ind) != len(indexing_axes):
        raise EindexError(f"Number of indexers {len(ind)},  expected {len(indexing_axes)}")

    for indexer in ind:
        # we only require indices to have the same dimensionality and being co-broadcastable
        expected_dimensionality = len(indexer_other_axes_names)
        indexer_dimensionality = len(indexer.shape)
        if expected_dimensionality != indexer_dimensionality:
            raise EindexError(
                f"All indexers should have {expected_dimensionality}, but found one with {indexer_dimensionality} "
            )

    flat_index = 0
    shift = 1
    # NB: traversing in reverse direction
    # this implementation (compared to simpler one) is more 'parallelizable' as sum of integers is associative
    for axis_name in flat_index_over[::-1]:
        if axis_name not in known_axes_sizes:
            raise EindexError(f"Size of axis {axis_name} was not inferred and should be specified")
        axis_len = known_axes_sizes[axis_name]
        if axis_name in indexing_axes:
            indexer = ind[indexing_axes.index(axis_name)]
            flat_index = flat_index + shift * (indexer % axis_len)
            shift *= axis_len
        else:
            axis_id = indexer_other_axes_names.index(axis_name)
            flat_index = (
                flat_index
                + ixp.arange_at_position(
                    len(indexer_other_axes_names),
                    axis=axis_id,
                    axis_len=axis_len,
                    array_to_copy_device_from=ind[0],
                )
                * shift
            )
            shift *= known_axes_sizes[axis_name]

    # here is the tricky part: we allow ind elements to have different shapes,
    # and we also allow them to be not used (_underscored indexers)
    # which means that result of arithmetic operations can still have some dimensions unexpanded
    broadcasted_shape = _broadcast_shapes([x.shape for x in ind])
    flat_index = ixp.xp.broadcast_to(flat_index, broadcasted_shape)

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
                raise EindexError(f"Wrong usage of indexer variable '{axis}' in {pattern}")
            elif presence == (True, True, False, True):
                self.batch_axes.append(axis)
            elif presence == (True, False, False, True):
                self.result_and_index_axes.append(axis)
            elif presence == (True, True, False, False):
                self.result_and_array_axes.append(axis)
            else:
                raise EindexError(f"Axis '{axis}' is used incorrectly in {pattern}")

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

    def apply_to_array_api(self, ixp: IXP, arr: T, ind: Union[T, List[T]]) -> T:
        known_axes_sizes: dict[str, int] = {}
        ind_list = _index_to_list_array_api(ind)

        for indexer in ind_list:
            assert len(indexer.shape) == len(self.pattern_parser.ind_other_axes_names)

        # step 1. transpose, reshapes of arr; learn its dimensions
        arr_2d = self.array_composition.compose_ixp(ixp, arr, known_axes_sizes)

        # step 2. compute shifts and create an actual indexing array
        full_index = compute_full_index_ixp(
            ixp,
            ind=ind_list,
            indexing_axes=self.pattern_parser.ind_axes_names,
            indexer_other_axes_names=self.pattern_parser.ind_other_axes_names,
            flat_index_over=self.batch_axes + self.pattern_parser.ind_axes_names,
            known_axes_sizes=known_axes_sizes,
        )

        # step 3. Flatten index
        full_index = self.index_composition.compose_ixp(ixp, full_index, known_axes_sizes)

        # step 4. indexing
        xp = ixp.xp
        # xp.take for 1d is implemented in the next version of numpy and cupy.
        result_2d = xp.take(arr_2d, full_index, axis=0)

        # step 5. reshape result to correct form
        return self.result_composition.decompose_ixp(ixp, result_2d, known_axes_sizes)

    def apply_to_numpy(self, ixp: IXP, arr: T, ind: Union[T, List[T]]) -> T:
        known_axes_sizes: dict[str, int] = {}
        ind_list = _index_to_list_array_api(ind)

        for indexer in ind_list:
            assert len(indexer.shape) == len(self.pattern_parser.ind_other_axes_names)

        # step 1. transpose, reshapes of arr; learn its dimensions
        arr_2d = self.array_composition.compose_ixp(ixp, arr, known_axes_sizes)

        # step 2. compute shifts and create an actual indexing array
        full_index = compute_full_index_ixp(
            ixp,
            ind=ind_list,
            indexing_axes=self.pattern_parser.ind_axes_names,
            indexer_other_axes_names=self.pattern_parser.ind_other_axes_names,
            flat_index_over=self.batch_axes + self.pattern_parser.ind_axes_names,
            known_axes_sizes=known_axes_sizes,
        )

        # step 3. Flatten index
        full_index = self.index_composition.compose_ixp(ixp, full_index, known_axes_sizes)

        # step 4. indexing
        import numpy as np

        result_2d = np.take(arr_2d, full_index, axis=0)

        # step 5. reshape result to correct form
        return self.result_composition.decompose_ixp(ixp, result_2d, known_axes_sizes)


class GatherFormula:
    def __init__(self, pattern: str, aggregation: Optional[Aggregation]) -> None:
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
                if aggregation is None:
                    raise EindexError(f"Axis '{axis}' can't be reduced as no aggregation set: {pattern}")
                self.array_and_index_axes.append(axis)
            elif presence == (False, False, False, True):
                if aggregation is None:
                    raise EindexError(f"Axis '{axis}' can't be reduced as no aggregation set: {pattern}")
                self.index_only_axes.append(axis)
            else:
                raise EindexError(f"Axis '{axis}' is used incorrectly in {pattern}")

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

    def apply_to_array_api(self, ixp: IXP, arr: T, ind: Union[T, List[T]]) -> T:
        known_axes_sizes: dict[str, int] = {}
        ind_list = _index_to_list_array_api(ind)

        for indexer in ind_list:
            assert len(indexer.shape) == len(self.parsed_pattern.ind_other_axes_names)

        # step 1. transpose, reshapes of arr; learn its dimensions
        arr_2d = self.array_composition.compose_ixp(ixp, arr, known_axes_sizes)

        # step 2. compute shifts and create an actual indexing array
        full_index_2d = compute_full_index_ixp(
            ixp,
            ind=ind_list,
            indexing_axes=self.parsed_pattern.ind_axes_names,
            indexer_other_axes_names=self.parsed_pattern.ind_other_axes_names,
            flat_index_over=self.index_walks,
            known_axes_sizes=known_axes_sizes,
        )

        # step 3. Flatten index
        full_index_2d = self.index_composition.compose_ixp(ixp, full_index_2d, known_axes_sizes)

        # step 4. indexing
        xp = ixp.xp
        # xp.take is implemented in numpy 1.25 and cupy
        # only 1d indexing is supported by xp.take, so we compose dims in index
        result_squashed = xp.take(arr_2d, xp.reshape(full_index_2d, (-1,)), axis=0)
        # and decompose dims in result
        arr_3d = xp.reshape(result_squashed, [*full_index_2d.shape, result_squashed.shape[-1]])

        if self.aggregation is None:
            assert arr_3d.shape[0] == 1
            result_2d = arr_3d[0, :, :]
        elif self.aggregation == "sum":
            result_2d = xp.sum(arr_3d, axis=0)
        elif self.aggregation == "min":
            result_2d = xp.min(arr_3d, axis=0)
        elif self.aggregation == "max":
            result_2d = xp.max(arr_3d, axis=0)
        elif self.aggregation == "mean":
            result_2d = xp.mean(arr_3d, axis=0)
        else:
            raise NotImplementedError(f"Reduction {self.aggregation} is not available")

        # step 5. reshape result to correct form
        return self.result_composition.decompose_ixp(ixp, result_2d, known_axes_sizes)

    def apply_to_numpy(self, ixp: IXP, arr, ind):
        known_axes_lengths: dict[str, int] = {}
        ind_list = _index_to_list_array_api(ind)

        for indexer in ind_list:
            assert len(indexer.shape) == len(self.parsed_pattern.ind_other_axes_names)

        # step 1. transpose, reshapes of arr; learn its dimensions
        arr_2d = self.array_composition.compose_ixp(ixp, arr, known_axes_lengths)

        # step 2. compute shifts and create an actual indexing array
        full_index_2d = compute_full_index_ixp(
            ixp,
            ind=ind_list,
            indexing_axes=self.parsed_pattern.ind_axes_names,
            indexer_other_axes_names=self.parsed_pattern.ind_other_axes_names,
            flat_index_over=self.index_walks,
            known_axes_sizes=known_axes_lengths,
        )

        # step 3. Flatten index
        full_index_2d = self.index_composition.compose_ixp(ixp, full_index_2d, known_axes_lengths)

        # step 4. indexing
        # cshape = self.result_composition.composed_shape
        # shape = [_prod(known_axes_lengths[var] for var in group) for group in cshape]
        if self.aggregation is None:
            assert full_index_2d.shape[0] == 1
            result_2d = arr_2d[full_index_2d[0]]
        elif self.aggregation == "sum":
            result_2d = arr_2d[full_index_2d].sum(axis=0)
        elif self.aggregation == "min":
            result_2d = arr_2d[full_index_2d].min(axis=0)
        elif self.aggregation == "max":
            result_2d = arr_2d[full_index_2d].max(axis=0)
        elif self.aggregation == "mean":
            result_2d = arr_2d[full_index_2d].mean(axis=0)
        else:
            raise NotImplementedError(f"Reduction {self.aggregation} is not available")

        # step 5. reshape result to correct form
        return self.result_composition.decompose_ixp(ixp, result_2d, known_axes_lengths)


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
                raise EindexError(f"Axis {axis} is used incorrectly in '{pattern}'")

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

    def apply_to_numpy(self, ixp: IXP, arr: T, ind: Union[T, List[T]], axis_sizes: dict[str, int]):
        ind_list = _index_to_list_array_api(ind)
        known_axes_lengths = {**axis_sizes}

        # step 0. reshape arr to [(b t) (c)]
        arr_2d = self.array_composition.compose_ixp(ixp, arr, known_axes_lengths=known_axes_lengths)

        # step 1. build first index of shape [b t s replica] -> (b h w)
        # some output axes may be present only in ind_other_axes
        index_other_shape = _broadcast_shapes([x.shape for x in ind_list])
        for axis_len, axis in zip(index_other_shape, self.parsed_pattern.ind_other_axes_names, strict=True):
            if axis not in known_axes_lengths:
                known_axes_lengths[axis] = axis_len
            else:
                assert axis_len == known_axes_lengths[axis]

        flat_index = compute_full_index_ixp(
            ixp,
            ind=ind_list,
            indexing_axes=self.parsed_pattern.ind_axes_names,
            indexer_other_axes_names=self.parsed_pattern.ind_other_axes_names,
            flat_index_over=self.index_walks,
            known_axes_sizes=known_axes_lengths,
        )

        # step 2. reshape flat index into [(s replica) (b t)] -> (b h w)
        flat_index_2d = self.index_composition.compose_ixp(ixp, flat_index, known_axes_lengths=known_axes_lengths)

        # step 3. creation of result in composed_shape
        cshape = self.result_composition.composed_shape
        shape = [_prod(known_axes_lengths[var] for var in group) for group in cshape]
        dtype = arr.dtype

        import numpy as np

        # step 4. aggregation
        if self.aggregation == "sum":
            result = np.zeros(shape, dtype=dtype)
            np.add.at(result, flat_index_2d, arr_2d)
        elif self.aggregation == "max":
            result = np.full(shape, fill_value=-np.inf, dtype=dtype)
            np.maximum.at(result, flat_index_2d, arr_2d)
        elif self.aggregation == "min":
            result = np.full(shape, fill_value=np.inf, dtype=dtype)
            np.minimum.at(result, flat_index_2d, arr_2d)
        elif self.aggregation == "mean":
            # mean is not ufunc and can't be just accumulated
            assert dtype in [np.float16, np.float32, np.float64], "mean reduction supported only for float tensors"
            nom = np.zeros(shape, dtype=dtype)
            np.add.at(nom, flat_index_2d, arr_2d)
            denom = np.zeros(shape, dtype=np.int64)
            np.add.at(denom, flat_index_2d, 1)
            result = nom / denom
            assert nom.shape == result.shape
        else:
            raise NotImplementedError(self.aggregation)

        return self.result_composition.decompose_ixp(ixp, result, known_axes_lengths=known_axes_lengths)


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
                raise EindexError(f"Axis {axis} is used incorrectly in '{pattern}'")

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

    def apply_to_numpy(self, ixp: IXP, arr: T, ind: Union[T, List[T]], axis_sizes: dict[str, int]):
        import numpy as np

        ind_list = _index_to_list_array_api(ind)
        known_axes_lengths = {**axis_sizes}

        # step 0. reshape arr to [(b h w) (c)]
        arr_2d = self.array_composition.compose_ixp(ixp, arr, known_axes_lengths=known_axes_lengths)

        # step 1. build first index of shape [b t order] -> (b h w)
        # some output axes may be present only in ind_other_axes
        index_other_shape = _broadcast_shapes([x.shape for x in ind_list])
        for axis_len, axis in zip(index_other_shape, self.parsed_pattern.ind_other_axes_names, strict=True):
            if axis not in known_axes_lengths:
                known_axes_lengths[axis] = axis_len
            else:
                assert axis_len == known_axes_lengths[axis]

        first_flat_index = compute_full_index_ixp(
            ixp,
            ind=ind_list,
            indexing_axes=self.parsed_pattern.ind_axes_names,
            indexer_other_axes_names=self.parsed_pattern.ind_other_axes_names,
            flat_index_over=self.index1_walks,
            known_axes_sizes=known_axes_lengths,
        )
        # step 2. and take elements into [(b t order) (c)]
        taken_2d = np.take(arr_2d, first_flat_index.flatten(), axis=0)

        # step 3. build second index of shape [b t order] -> (b t h w2), put elements into [(b t h w2) (c)]
        flat_index_axes = self.index2_walks
        second_flat_index = compute_full_index_ixp(
            ixp,
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

        dtype = arr.dtype

        if self.aggregation == "sum":
            result_2d = np.zeros([first_axis, taken_2d.shape[1]], dtype=dtype)
            np.add.at(result_2d, second_flat_index.flatten(), taken_2d)
        elif self.aggregation == "max":
            result_2d = np.full([first_axis, taken_2d.shape[1]], fill_value=-np.inf, dtype=dtype)
            np.maximum.at(result_2d, second_flat_index.flatten(), taken_2d)
        elif self.aggregation == "min":
            result_2d = np.full([first_axis, taken_2d.shape[1]], fill_value=np.inf, dtype=dtype)
            np.minimum.at(result_2d, second_flat_index.flatten(), taken_2d)
        elif self.aggregation == "mean":
            assert dtype in [np.float16, np.float32, np.float64], "Mean-reduction supported only for floating dtypes"
            result_2d_nom = np.zeros([first_axis, taken_2d.shape[1]], dtype=dtype)
            np.add.at(result_2d_nom, second_flat_index.flatten(), taken_2d)
            result_2d_denom = np.zeros([first_axis, taken_2d.shape[1]], dtype=dtype)
            np.add.at(result_2d_denom, second_flat_index.flatten(), 1)
            result_2d = result_2d_nom / result_2d_denom
            assert result_2d.shape == result_2d_nom.shape
        else:
            raise NotImplementedError(f"Unknown reduction: {self.aggregation}")

        # step 5
        return self.result_decomposition.decompose_ixp(ixp, result_2d, known_axes_lengths=known_axes_lengths)


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
            raise EindexError("At least one indexing axis should be in [...]")

        double_used_axes = set.intersection(set(self.indexing_axes), set(self.indexing_other_axes))
        if double_used_axes:
            raise EindexError(f"Some axes were used more than once: {double_used_axes}")

        on_one_side = set.symmetric_difference(
            set(self.input_axes), set.union(set(self.indexing_axes), set(self.indexing_other_axes))
        )
        if on_one_side:
            raise EindexError(f"All axes should be present in left and right side, but these are not: {on_one_side}")

        self.transposition: list[int] = [self.input_axes.index(axis) for axis in self.indexing_other_axes]
        # note: we place indexing axes in reverse order here
        self.transposition += [self.input_axes.index(axis) for axis in self.indexing_axes[::-1]]
        self.is_max: bool = is_max

    def apply_to_ixp(self, ixp: IXP, arr):
        arr = ixp.permute_dims(arr, self.transposition)
        xp = ixp.xp

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
            raise EindexError(f"Special axis {order_axis} should not be in the left part of pattern: {pattern}")
        if order_axis not in self.indexing_other_axes:
            raise EindexError(f"Special axis {order_axis} should be in the result part outside of [...]: {pattern}")

        repeated_axes = set.intersection(set(self.indexing_axes), set(self.indexing_other_axes))
        if repeated_axes:
            raise EindexError(f"Axes {repeated_axes} were repeated in result part: {pattern} ")

        difference = set.symmetric_difference(
            {*self.input_axes, order_axis},
            {*self.indexing_axes, *self.indexing_other_axes},
        )
        if difference:
            raise EindexError(f"Axes {difference} should be present both in input and result of {pattern}")

        self.transposition: list[int] = []
        self.position_of_order_axis = self.indexing_other_axes.index(order_axis)
        for axis in self.indexing_other_axes:
            if axis == order_axis:
                # note: we place indexing axes in reverse order here
                self.transposition += [self.input_axes.index(axis) for axis in self.indexing_axes[::-1]]
            else:
                self.transposition.append(self.input_axes.index(axis))

    def apply_to_ixp(self, ixp: IXP, arr):
        xp = ixp.xp
        arr = ixp.permute_dims(arr, self.transposition)
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
