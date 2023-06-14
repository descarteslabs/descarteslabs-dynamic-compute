from __future__ import annotations

import dataclasses
from collections import namedtuple
from collections.abc import Generator
from copy import deepcopy
from typing import Callable, Union

from descarteslabs.geo import AOI

from .compute_map import ComputeMap
from .operations import compute_aoi, set_cache_id
from .serialization import BaseSerializationModel

BUILT_IN_REDUCERS = ["max", "min", "mean", "median", "sum", "std"]

ImageStackReducer = namedtuple("ImageStackReducer", ["func", "axis"])


class ImageStackGroups:
    def __init__(self, image_stack, groups_graft):
        self.groups_graft = groups_graft
        self.image_stack = image_stack
        self.computed_value = None
        self.computed_AOI: AOI = None
        self.reducer: ImageStackReducer = None

    def compute(self, aoi: AOI) -> Generator:
        if not self.computed_value or self.computed_AOI != aoi:
            self.computed_value, _ = compute_aoi(self.groups_graft, aoi)
            self.computed_AOI = aoi
        for group_name, id_list in self.computed_value:
            image_stack = self.image_stack.filter(lambda i: i.id in id_list)
            if (
                self.reducer
                and isinstance(self.reducer.func, str)
                and self.reducer.func in BUILT_IN_REDUCERS
            ):
                image_stack = getattr(image_stack, self.reducer.func)(
                    axis=self.reducer.axis
                )
            elif self.reducer and isinstance(self.reducer.func, Callable):
                image_stack = image_stack.reduce(self.reducer.func, self.reducer.axis)
            yield group_name, image_stack


@dataclasses.dataclass
class ImageStackGroupBySerializationModel(BaseSerializationModel):
    image_stack_json: str
    groups_graft: dict
    reducer: ImageStackReducer = None


class ImageStackGroupBy(ComputeMap):
    """

    ImageStackGroupBy class offers various methods to work with groupings computed via ImageStack.groupby

    """

    def __init__(self, image_stack, groups_graft):
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

        Parameters:
            axis (str, optional): _description_. Defaults to "images".

        Returns:
            ImageStackGroups object
        """

        return self.map("max", axis)

    def min(self, axis="images"):
        """
        Performs a min operation on every grouped ImageStack

        Parameters:
            axis (str, optional): _description_. Defaults to "images".

        Returns:
            ImageStackGroups object
        """

        return self.map("min", axis)

    def median(self, axis="images"):
        """
        Performs a median operation on every grouped ImageStack

        Parameters:
            axis (str, optional): _description_. Defaults to "images".

        Returns:
            ImageStackGroups object
        """

        return self.map("median", axis)

    def mean(self, axis="images"):
        """
        Performs a mean reduction operation on every grouped ImageStack

        Parameters:
            axis (str, optional): _description_. Defaults to "images".

        Returns:
            ImageStackGroups object
        """

        return self.map("mean", axis)

    def sum(self, axis="images"):
        """
        Performs a sum reduction operation on every grouped ImageStack

        Parameters:
            axis (str, optional): _description_. Defaults to "images".

        Returns:
            ImageStackGroups object
        """
        return self.map("sum", axis)

    def std(self, axis="images"):
        """
        Performs an std reduction operation on every grouped ImageStack

        Parameters:
            axis (str, optional): _description_. Defaults to "images".

        Returns:
            ImageStackGroups object
        """

        return self.map("std", axis)

    def map(self, function: Union[str, Callable], axis: str = "images"):
        """
        Applies a function to each grouped ImageStack. `function` can be a string
        of ["max", "min", "mean", "median", "sum", "std"], or a callable function.

        Args:
            function (Union[str, Callable]): str or Function to apply to each grouped ImageStack. If str must be
                                             one of ["max", "min", "mean", "median", "sum", "std"]
            axis (str): one of ["pixels", "images", "bands"]

        Raises:
            TypeError: if function_name not a member of ["max", "min", "mean", "median", "sum", "std"]

        Returns:
            ImageStackGroups object
        """
        if isinstance(function, str) and function not in BUILT_IN_REDUCERS:
            raise TypeError(
                f"Reducer {function} is not a member of {BUILT_IN_REDUCERS}"
            )
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
