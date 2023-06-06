import json
from collections import namedtuple
from collections.abc import Generator
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Type, Union

from descarteslabs.geo import AOI

from .blob import GRAFTS_NAMESPACE, create_blob_and_upload_data, load_graft_from_blob
from .compute_map import ComputeMap
from .operations import compute_aoi, set_cache_id

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
        groups = deepcopy(self.groups)
        groups.reducer = ImageStackReducer(function, axis)
        return groups

    def save_to_catalog_blob(
        self,
        name: str,
        description: Optional[str] = None,
        extra_properties: Optional[Dict[str, Union[str, int, float]]] = None,
        readers: Optional[List[str]] = None,
        writers: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Saves this object to catalog as a Blob

        Parameters
        ----------
        name : str
            The name to give the blob in catalog
        description : Optional[str], optional
            A description of the blob, by default None
        extra_properties : Optional[dict[str, Union[str, int, float]]], optional
            Any extra properties to be stored in the blob, by default None
        readers : Optional[list[str]], optional
            A list of emails, orgs, groups, and users to give read access to the blob, by default None
        writers : Optional[list[str]], optional
            A list of emails, orgs, groups, and users to give write access to the blob, by default None
        tags : Optional[List[str]], optional
            A list of tags to assign to the blob

        Returns
        -------
        str
            The id of the blob created
        """
        extra_properties = extra_properties or {}
        extra_properties["graft_type"] = self.__class__.__name__

        blob = create_blob_and_upload_data(
            json.dumps(
                {
                    "image_stack": self.image_stack,
                    "groups_graft": self.groups_graft,
                    "scenes_graft": self.image_stack.scenes_graft,
                    "bands": self.image_stack.bands,
                    "product_id": self.image_stack.product_id,
                }
            ),
            name,
            namespace=GRAFTS_NAMESPACE,
            description=description,
            extra_properties=extra_properties,
            readers=readers,
            writers=writers,
            tags=tags,
        )

        return blob.id

    @classmethod
    def load_from_catalog_blob(cls, name: str) -> Type[ComputeMap]:
        """Loads an dynamic compute type from catalog

        Parameters
        ----------
        name : str
            The name of the blob in catalog

        Returns
        -------
        Type[ComputeMap]
            The loaded object
        """

        graft_dict = load_graft_from_blob(name, cls.__name__)
        image_stack = cls.__SUBCLASSES__["ImageStack"](
            graft_dict["image_stack"],
            graft_dict["scenes_graft"],
            graft_dict["bands"],
            graft_dict["product_id"],
        )

        return cls(image_stack, graft_dict["groups_graft"])
