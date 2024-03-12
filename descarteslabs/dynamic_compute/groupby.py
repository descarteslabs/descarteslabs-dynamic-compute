from __future__ import annotations

import dataclasses
from collections import namedtuple
from collections.abc import Generator
from copy import deepcopy
from typing import Any, Callable, Hashable, Union

from descarteslabs.geo import AOI
from tqdm import tqdm

from .compute_map import ComputeMap
from .image_stack import ImageStack
from .operations import compute_aoi, encode_function, groupby, reset_graft, set_cache_id
from .reductions import reduction
from .serialization import BaseSerializationModel

BUILT_IN_REDUCERS = ["max", "min", "mean", "median", "sum", "std"]

ImageStackReducer = namedtuple("ImageStackReducer", ["func", "axis"])


class ImageStackGroups:
    def __init__(self, image_stack: ImageStack, groups_graft: dict):
        self.groups_graft = groups_graft
        self.image_stack = image_stack
        self.computed_value = None
        self.computed_AOI: AOI = None
        self.reducer: ImageStackReducer = None

    def compute(self, aoi: AOI) -> Generator[Any, ComputeMap]:
        """
        Evaluate this groups object for a particular AOI and return a generator yielding
        tupleS of (group key, ComputeMap)

        Parameters
        ----------
            aoi (AOI): descarteslabs.geo.GeoContext
        GeoContext for which to compute evaluate these groups

        Returns
        -------
            Generator[Any, ComputeMap]: generator which yields a tuple: (group key, ComputeMap)
        """

        if not self.computed_value or self.computed_AOI != aoi:
            self.computed_value, _ = compute_aoi(self.groups_graft, aoi)
            self.computed_AOI = aoi

        for group_name, id_list in self.computed_value:
            compute_map = self.image_stack.filter(lambda i: i.id in id_list)
            if (
                self.reducer
                and isinstance(self.reducer.func, str)
                and self.reducer.func in BUILT_IN_REDUCERS
            ):
                compute_map = getattr(compute_map, self.reducer.func)(
                    axis=self.reducer.axis
                )
            elif (
                self.reducer
                and isinstance(self.reducer.func, str)
                and self.reducer.func not in BUILT_IN_REDUCERS
            ):

                compute_map = reduction(
                    self.image_stack, self.reducer.func, axis=self.reducer.axis
                )
            yield group_name, compute_map

    def compute_all(self, aoi: AOI) -> dict:
        """
        Compute the groups for `aoi`, compute each resulting key/ComputeMap pair for the supplied AOI,
        and return a dictionary containing a key for each unique group and its computed DotDict. This
        function, potentially long running, features a progressbar display.

        Parameters
        ----------
            aoi (AOI): descarteslabs.geo.GeoContext
        GeoContext for which to compute evaluate these groups

        Returns
        -------
            dict: {group key: DotDict, ...} A Dict, where the keys are the computed group keys and the
            values are the computed DotDict corresponding to each key
        """

        uncomputed_groups = list(self.compute(aoi))
        pbar = tqdm(
            uncomputed_groups,
            desc="Processing groups",
            bar_format="{desc:<18}{percentage:3.0f}%|{bar:10}{r_bar}",
            unit=" groups",
        )
        computed_groups = {group_id: value.compute(aoi) for group_id, value in pbar}
        return computed_groups

    def one(self, aoi: AOI) -> tuple:
        """
        A Tuple of (group key, DotDict) for one random group. Helpful for debugging.

        Parameters
        ----------
            aoi (AOI): descarteslabs.geo.GeoContext
        GeoContext for which to compute evaluate these groups

        Returns
        -------
            tuple: (group key, DotDict) A tuple of a single group key and its corresponding computed DotDict
        """
        group_name, value = next(self.compute(aoi))
        return group_name, value.compute(aoi)


@dataclasses.dataclass
class ImageStackGroupBySerializationModel(BaseSerializationModel):
    image_stack_json: str
    groups_graft: dict
    reducer: ImageStackReducer = None

    @classmethod
    def from_json(cls, data: str) -> ImageStackGroupBySerializationModel:
        base_obj = super().from_json(data)
        base_obj.groups_graft = reset_graft(base_obj.groups_graft)
        return base_obj


class ImageStackGroupBy(ComputeMap):
    """

    ImageStackGroupBy class offers various methods to work with groupings computed via ImageStack.groupby

    """

    def __init__(self, image_stack: ImageStack, groups_graft: dict):
        set_cache_id(groups_graft)
        super().__init__(groups_graft)
        self.image_stack = image_stack
        self.groups_graft = groups_graft
        self.groups = ImageStackGroups(image_stack, groups_graft)

    def compute(self, *args, **kwargs):
        raise Exception(
            "ImageStackGroupBy cannot be computed directly. "
            "Instead, compute `.groups`, use `.max` or `.mean`, etc. to "
            "composite the groups into a single ImageStack and compute that "
        )

    def max(self, axis="images"):
        """Performs a max operation on every grouped ImageStack

        Parameters
        ----------
            axis (str, optional): _description_. Defaults to "images".

        Returns
        -------
            ImageStackGroups object
        """

        return self.map("max", axis)

    def min(self, axis="images"):
        """
        Performs a min operation on every grouped ImageStack

        Parameters
        ----------
            axis (str, optional): _description_. Defaults to "images".

        Returns
        -------
            ImageStackGroups object
        """

        return self.map("min", axis)

    def median(self, axis="images"):
        """
        Performs a median operation on every grouped ImageStack

        Parameters
        ----------
            axis (str, optional): _description_. Defaults to "images".

        Returns
        -------
            ImageStackGroups object
        """

        return self.map("median", axis)

    def mean(self, axis="images"):
        """
        Performs a mean reduction operation on every grouped ImageStack

        Parameters
        ----------
            axis (str, optional): _description_. Defaults to "images".

        Returns
        -------
            ImageStackGroups object
        """

        return self.map("mean", axis)

    def sum(self, axis="images"):
        """
        Performs a sum reduction operation on every grouped ImageStack

        Parameters
        ----------
            axis (str, optional): _description_. Defaults to "images".

        Returns
        -------
            ImageStackGroups object
        """
        return self.map("sum", axis)

    def std(self, axis="images"):
        """
        Performs an std reduction operation on every grouped ImageStack

        Parameters
        ----------
            axis (str, optional): _description_. Defaults to "images".

        Returns
        -------
            ImageStackGroups object
        """

        return self.map("std", axis)

    def map(self, function: Union[str, Callable], axis: str = "images"):
        """
        Applies a function to each grouped ImageStack. `function` can be a string
        of ["max", "min", "mean", "median", "sum", "std"], or a callable function.

        Parameters
        ----------
            function (Union[str, Callable]): str or Function to apply to each grouped ImageStack. If str must be
                                             one of ["max", "min", "mean", "median", "sum", "std"]
            axis (str): one of ["pixels", "images", "bands"]

        Raises
        ------
            TypeError: if function_name not a member of ["max", "min", "mean", "median", "sum", "std"]

        Returns
        -------
            ImageStackGroups object
        """

        if isinstance(function, str) and function not in BUILT_IN_REDUCERS:
            raise TypeError(
                f"Reducer {function} is not a member of {BUILT_IN_REDUCERS}"
            )

        elif isinstance(function, Callable):
            function = encode_function(function)

        new_imagestackgroupby = ImageStackGroupBy(self.image_stack, self.groups_graft)
        groups = deepcopy(self.groups)
        groups.reducer = ImageStackReducer(function, axis)

        new_imagestackgroupby.groups = groups

        return new_imagestackgroupby

    def serialize(self):
        """Serializes this object into a json representation"""

        return ImageStackGroupBySerializationModel(
            image_stack_json=self.image_stack.serialize(),
            groups_graft=self.groups_graft,
            reducer=self.groups.reducer,
        ).json()

    @classmethod
    def deserialize(cls, data: str) -> ImageStackGroupBy:
        """Deserializes into this object from json

        Parameters
        ----------
        data : str
            The json representation of the object state

        Returns
        -------
        ImageStackGroupby
            An instance of this object with the state stored in data
        """

        model = ImageStackGroupBySerializationModel.from_json(data)

        imagestack_groupby = cls(
            ComputeMap.__SUBCLASSES__["ImageStack"].deserialize(model.image_stack_json),
            model.groups_graft,
        )
        imagestack_groupby.groups.reducer = (
            ImageStackReducer(*model.reducer) if model.reducer else None
        )
        return imagestack_groupby


def image_stack_groupby(
    image_stack: ImageStack, grouping_func: Callable[[dict], Hashable]
) -> ImageStackGroupBy:
    """
    Perform a grouping function over either images or bands and return an ImageStackGroupBy object.

    Parameters
    ----------
    grouping_func: Callable[[dict], Hashable]
        Function to pick out the values to group by. The function should take as
        input a dotted-dict representing metadata for an image and should return a
        hashable value upon which groups will be build.

    Returns
    -------
    ImageStackGroupBy object.

    Example
    -------
    >>> import descarteslabs.dynamic_compute as dc # doctest: +SKIP
    >>> m = dc.map # doctest: +SKIP
    >>> m # doctest: +SKIP
    >>> sigma0_vv = dc.ImageStack.from_product_bands( # doctest: +SKIP
            "esa:sentinel-1:sigma0v:v1", "vv", "20230101", "20230401" # doctest: +SKIP
        ) # doctest: +SKIP
    >>> # group by acquired month
    >>> grouped_sigma = sigma0_vv.groupby(lambda x: x.acquired.month) # doctest: +SKIP
    >>> # loop through each grouping, applying a max reducer and visualize it on the map
    >>> for group_name, image_stack in grouped_sigma.groups.compute(m.geocontext()): # doctest: +SKIP
            image_stack.max(axis="images").visualize(str(group_name), m, colormap="turbo") # doctest: +SKIP

    """
    encoded_grouping_func = encode_function(grouping_func)
    groups = groupby(dict(image_stack), encoded_grouping_func)

    return ImageStackGroupBy(image_stack, groups)


ImageStack.groupby = image_stack_groupby
