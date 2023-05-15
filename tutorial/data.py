from typing import Dict, List, Optional, Union

import numpy as np


def raw_representation(x):
    if x.dtype in ["float16", "float32", "float64"]:
        return f"{float(x):.3g}"
    else:
        return str(x)


class DemoData:
    def __init__(self):
        self.heroes = [
            "Yoda",
            "Alice",
            "Gandalf",
            "R. Daneel Olivaw",
            "Devola & Popola",
            # "Harry James Potter-Evans-Verres",
            # Aerith,
            # Aloy,
        ]

        self.worlds = [
            "Amber",
            "Beauxbatons",
            "Emerald\u00A0City",
            # 'Trantor',
            "Midgar",
            "Mycogen",
        ]

        self.places = [
            "City\u00A0Center",
            "Palace",
            "University",
            "Forest\u00A0Camp",
            # 'Old\u00A0City',
        ]

        self.years = [
            2020,
            2021,
        ]

        self.months = [
            "August",
            "September",
            "October",
            "Neverbruary",
        ]

        self.networks = [
            # 'FriendFleet',
            "CharmHouse",
            "WitchedIn",
            "AdScroll",
        ]

        self.n_hero = len(self.heroes)
        self.n_world = len(self.worlds)
        self.n_place = len(self.places)
        self.n_year = len(self.years)
        self.n_month = len(self.months)
        self.n_network = len(self.networks)

        rng = np.random.RandomState(42)

        self.temp = rng.uniform(-20, 40, size=[self.n_world, self.n_place, self.n_year, self.n_month])
        self.visitors = (2 ** rng.uniform(8, 20, size=[self.n_world, self.n_place, self.n_year, self.n_month])).astype(
            "int32"
        )
        self.photos = rng.randint(0, 20, size=[self.n_hero, self.n_year, self.n_month])

        travels_world = rng.randint(0, self.n_world, size=[self.n_hero, self.n_year, self.n_month])
        travels_place = rng.randint(0, self.n_place, size=[self.n_hero, self.n_year, self.n_month])
        self.travels = np.asarray([travels_world, travels_place])

        n_subscriptions = self.n_hero * self.n_year
        self.subscriptions = np.asarray(
            [
                rng.randint(0, self.n_network, n_subscriptions),
                rng.randint(0, self.n_hero, n_subscriptions),
                rng.randint(0, self.n_year, n_subscriptions),
            ]
        )  # [network, hero, year] subscription

    def _get_axis(self, axis_name: str, additional_axes: Dict[str, list]):
        # allow overriding if necessary
        if axis_name in additional_axes:
            return additional_axes[axis_name]

        # allow having names like world_to to be recognized as world
        axis_name = axis_name.split("_")[0]

        if axis_name == "hero":
            return self.heroes
        if axis_name == "world":
            return self.worlds
        if axis_name == "place":
            return self.places
        if axis_name == "year":
            return self.years
        if axis_name == "month":
            return self.months
        if axis_name == "network":
            return self.networks

        raise ValueError(axis_name)

    def visualize(
        self,
        tensor: Union[list, np.ndarray],
        index_pattern: str,
        cols: str,
        rows: str | None = None,
        additional_axes: Optional[dict] = None,
    ):
        if additional_axes is None:
            additional_axes = {}
        from IPython.display import HTML, display  # type: ignore

        from eindex._parsing import _parse_indexing_part, _parse_space_separated_dimensions

        main_axes, other_axes = _parse_indexing_part(index_pattern)
        expected_shape = (len(main_axes), *[len(self._get_axis(n, additional_axes)) for n in other_axes])

        tensor = np.asarray(tensor)
        if tensor.shape != expected_shape:
            raise ValueError(f"Expected a tensor of dimensionality {expected_shape}, but got {tensor.shape}")

        tensor = list(tensor)
        assert len(tensor) == len(main_axes), "wrong pattern"
        assert tensor[0].ndim == len(other_axes)

        xs = _parse_space_separated_dimensions(cols)
        for x in xs:
            assert x in other_axes, (x, other_axes)
        if rows is None:
            ys = [x for x in other_axes if x not in cols]
        else:
            ys = _parse_space_separated_dimensions(rows)
        assert {*xs, *ys} == {*other_axes}

        def enumerate_multiaxis(dims: List[str]):
            axis_values = [
                [(dim, i, val) for i, val in enumerate(self._get_axis(dim, additional_axes))] for dim in dims
            ]
            from itertools import product

            for i, combination in enumerate(product(*axis_values)):
                axis2index_dict = {dim: i for (dim, i, _val) in combination}
                yield i, axis2index_dict, [str(val) for _dim, _i, val in combination]

        def values_at(**values_dict):
            index = tuple(values_dict[axis] for axis in other_axes)
            return [t[index] for t in tensor]

        def br_join(arr):
            return "<br />\n ".join(arr)

        elements_rowlabels = [
            f"<div style='grid-column: colname / span 1 ; grid-row: row{row_id} / span 1;'>{br_join(row_labels)}</div>"
            for row_id, _axes_dict, row_labels in enumerate_multiaxis(ys)
        ]

        elements_collabels = [
            f"<div style='grid-column: col{col_id} / span 1 ; grid-row: rowname / span 1;'>{br_join(col_labels)}</div>"
            for col_id, _axes_dict, col_labels in enumerate_multiaxis(xs)
        ]

        def decode(axis_name, val):
            if axis_name == "temp":
                return f"{val:.1f}"
            if axis_name == "other":
                return raw_representation(val)
            return self._get_axis(axis_name, additional_axes)[val]

        elements = []
        for row_id, row_axes_dict, _row_labels in enumerate_multiaxis(ys):
            for col_id, col_axes_dict, _col_labels in enumerate_multiaxis(xs):
                values = values_at(**row_axes_dict, **col_axes_dict)
                decoded = [decode(name, val) for name, val in zip(main_axes, values)]
                text = ", ".join(str(x) for x in decoded)
                is_number = text.removeprefix("-").replace(".", "", 1).isdigit()
                if is_number:
                    alignment = " text-align: right;"
                else:
                    alignment = ""

                elements += [
                    f"<div style='grid-column: col{col_id} / span 1 ; grid-row: row{row_id} / span 1; {alignment}'>{text}</div>"
                ]

        # ugly css. Yes, we can

        el_text = "\n".join([*elements_rowlabels, *elements_collabels, *elements])
        result_html = f"""
        <div style="
            display: inline-grid; 
            grid-template-columns: [colname] minmax(150px, 200px) { ' 1fr '.join(f'[col{i}]' for i in range(len(elements_collabels) + 1) )} ;
            grid-template-rows:    [rowname]  35px { ' 1fr '.join(f'[row{i}]' for i in range(len(elements_rowlabels) + 1) )} ;
            column-gap: 10px;
            row-gap: 5px;
            "
        >
            {el_text}
        </div>
        """

        # print(result_html)
        display(HTML(result_html))
