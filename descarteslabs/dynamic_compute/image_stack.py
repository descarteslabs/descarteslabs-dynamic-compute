from __future__ import annotations

import dataclasses
import datetime
import json
from typing import Callable, Dict, List, Optional, Tuple, Union

import descarteslabs as dl
import numpy as np

from .compute_map import (
    AddMixin,
    CompareMixin,
    ComputeMap,
    ExpMixin,
    FloorDivMixin,
    MulMixin,
    NumpyReductionMixin,
    SignedMixin,
    SubMixin,
    TrueDivMixin,
    as_compute_map,
)
from .datetime_utils import normalize_datetime, normalize_datetime_or_none
from .dl_utils import get_product_or_fail
from .groupby import ImageStackGroupBy
from .mosaic import Mosaic
from .operations import (
    _apply_binary,
    _concat_bands,
    _index,
    _length,
    _pick_bands,
    _rename_bands,
    adaptive_mask,
    encode_function,
    filter_scenes,
    format_bands,
    groupby,
    select_scenes,
    set_cache_id,
    stack_scenes,
)
from .reductions import reduction
from .serialization import BaseSerializationModel

AXIS_NAME_TO_INDEX_MAP = {"images": (0,), "bands": (1,), "pixels": (2, 3)}


@dataclasses.dataclass
class ImageStackSerializationModel(BaseSerializationModel):
    """State representation of a ImageStack instance"""

    full_graft: Dict
    scenes_graft: Dict
    product_id: str
    bands: Union[str, List[str]]
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None


class ImageStack(
    AddMixin,
    SubMixin,
    MulMixin,
    TrueDivMixin,
    FloorDivMixin,
    SignedMixin,
    ExpMixin,
    CompareMixin,
    NumpyReductionMixin,
    ComputeMap,
):
    # This class acts as a proxy for ImageCollections. The steps in evaluating
    # this object are
    # 1. Select scenes
    # 2. Optionally filter scenes
    # 3. Download scenes into an array.

    # Steps 1 and 2 are handled by a "scenes" graft that evaulates to an
    # ImageCollection, not an array. ImageCollections contain the metadata
    # necessary for filtering, which we want to do *before* downloading an array.
    #  This allows us to compose filtering operations by creating a new "scenes"
    #  graft that applies a new filtering operation to the existing "scenes" graft.
    #
    # The full graft uses the scenes graft to generate an ImageCollection and adds a instructions
    # to accesss the raster data.

    _RETURN_PRECEDENCE = 2

    def __init__(
        self,
        full_graft: Dict,
        scenes_graft: Optional[Dict] = None,
        bands: Optional[Union[str, List[str]]] = None,
        product_id: Optional[str] = None,
        start_datetime: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        end_datetime: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    ):
        """
        Initialize a new instance of ImageStack. Users should rely on
        from_product_bands

        Parameters
        ----------
        full_graft: Dict
            Graft, which when evaluated will generate a scenes x bands x rows
            x cols array
        scenes_graft: Dict
            Graft, which when evaluated will generate an ImageCollection object
        bands: Union[str, List[str]]
            Bands either as space separated names in a single string or a list
            of band names
        product_id: str
            Product id
        start_datetime: Optional[Union[str, datetime.date, datetime.datetime]]
            Optional initial cutoff for an ImageStack
        end_datetime: Optional[Union[str, datetime.date, datetime.datetime]]
            Optional final cutoff for an ImageStack
        """

        set_cache_id(full_graft)
        super().__init__(full_graft)
        self.scenes_graft = scenes_graft
        self.bands = bands
        self.product_id = str(product_id)
        self.start_datetime = normalize_datetime_or_none(start_datetime)
        self.end_datetime = normalize_datetime_or_none(end_datetime)
        self.init_args = {
            "scenes_graft": self.scenes_graft,
            "bands": self.bands,
            "product_id": self.product_id,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
        }

    @classmethod
    def from_product_bands(
        cls,
        product_id: str,
        bands: Union[str, List[str]],
        start_datetime: Union[str, datetime.date, datetime.datetime],
        end_datetime: Union[str, datetime.date, datetime.datetime],
        **kwargs,
    ) -> ImageStack:
        """
        Create a new ImageStack object

        Parameters
        ----------
        product_id: str
            ID of the product from which we want to access data
        bands: Union[str, List[str]]
            A space-separated list of bands within the product, or a list of strings.
        start_datetime: Union[str, datetime.date, datetime.datetime]
            Start date for image stack
        end_datetime: Union[str, datetime.date, datetime.datetime]
            End date for image stack

        Returns
        -------
        m: ImageStack
            New ImageStack object.
        """

        _ = get_product_or_fail(product_id)
        start_datetime = normalize_datetime(start_datetime)
        end_datetime = normalize_datetime(end_datetime)

        formatted_bands = " ".join(format_bands(bands))
        scenes_graft = select_scenes(
            product_id, formatted_bands, start_datetime, end_datetime
        )

        return cls(
            stack_scenes(scenes_graft, bands),
            scenes_graft,
            bands,
            product_id,
            start_datetime,
            end_datetime,
        )

    def filter(self, f: Callable[[dl.catalog.Image], bool]) -> ImageStack:
        """
        Filter an image stack to based on image properties.

        Parameters
        ----------
        f: Callable[[dl.catalog.Image], bool]
            Filter function. This function must take a dl.catalog.Image object and return
            a bool indicating that the image should be retained (True) or excluded (False)

        Returns
        -------
        ImageStack
            New ImageStack object.
        """

        if self.scenes_graft is None:
            raise Exception(
                "This ImageStack cannot be filtered because "
                "it no longer has image metadata. This can happen "
                "when, e.g., an ImageStack is created from a mathematical "
                "operation "
            )

        new_scenes_graft = filter_scenes(self.scenes_graft, encode_function(f))

        return ImageStack(
            stack_scenes(new_scenes_graft, self.bands),
            scenes_graft=new_scenes_graft,
            bands=self.bands,
            product_id=self.product_id,
        )

    def get(self, idx: int) -> Mosaic:
        """
        Access an image within the stack as a Mosaic. Note __getattr__ would be
        an arguably better function name, however this class inherits from dict.

        Parameters
        ----------
        idx: int
            Index to access

        Returns
        -------
        mosaic: Mosaic
            Mosaic proxy object associated with the index.
        """
        if not isinstance(idx, int):
            raise Exception("Index must be an integer")

        return Mosaic(_index(idx, self))

    def length(self):
        """
        Proxy object for the length of this image stack

        Returns
        -------
        compute_map: ComputeMap
            Proxy object for the length
        """
        return as_compute_map(_length(self))

    def pick_bands(self, bands: Union[str, List[str]]) -> ImageStack:
        """
        Create a new ImageStack object with the specified bands and
        the product-id of this ImageStack object

        Parameters
        ----------
        bands: str
            A space-separated list of bands within the product, or a list
            of bands as strings

        Returns
        -------
        m: ImageStack
            New mosaic object.
        """

        return ImageStack(
            _pick_bands(self, json.dumps(format_bands(bands))),
            scenes_graft=self.scenes_graft,
            bands=self.bands,
            product_id=self.product_id,
        )

    def unpack_bands(
        self, bands: Union[str, List[str]]
    ) -> Union[ImageStack, Tuple[ImageStack, ...]]:
        """
        Create a tuple of new ImageStack objects with one for each bands.

        Parameters
        ----------
        bands: str
            A space-separated list of bands within the product.

        Returns
        -------
        m: Tuple[ImageStack, ...]
            New mosaic object per band passed in.
        """

        bands = format_bands(bands)

        if len(bands) > 1:
            return tuple([self.pick_bands([band]) for band in bands])
        else:
            return self.pick_bands([bands[0]])

    def rename_bands(self, bands):
        """Rename the bands of an array."""

        return ImageStack(
            _rename_bands(self, json.dumps(format_bands(bands))),
            scenes_graft=self.scenes_graft,
            bands=self.bands,
            product_id=self.product_id,
        )

    def concat_bands(self, other: ImageStack) -> ImageStack:
        """
        Create a new ImageStack that stacks bands. This call does not
        mutate `this`

        Note that per-image metadata for the returned ImageStack instance is
        taken from `self` not `other`.

        Parameters
        ----------
        other: ImageStack
            concat the bands of this ImageStack with those of other

        Returns
        -------
        stack: ImageStack
            ImageStack object with stacked bands
        """

        if not isinstance(other, ImageStack):
            other = self.pick_bands(other)

        return ImageStack(
            _concat_bands(self, other),
            scenes_graft=self.scenes_graft,
        )

    def mask(self, mask: ComputeMap) -> ImageStack:
        """
        Apply a mask as a delayed object. This call does not
        mutate `this`

        Parameters
        ----------
        mask: ComputeMap
            Delayed object to use as a mask.

        Returns
        -------
        masked: Mosaic
            Masked mosaic.
        """

        return ImageStack(
            _apply_binary(mask, self, adaptive_mask, lambda pa, pb, **kwargs: pb),
            scenes_graft=self.scenes_graft,
            bands=self.bands,
            product_id=self.product_id,
        )

    def groupby(
        self: ImageStack, grouping_func: Callable[[np.ndarray], np.ndarray]
    ) -> ImageStackGroupBy:
        """
        Perform a grouping function over either images or bands and return an ImageStackGroupBy object.

        Parameters
        ----------
        grouping_func: Callable[[np.ndarray], np.ndarray]
            Function to pick out the values to group by

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
        >>> for group_name, image_stack in grouped_sigma.compute(m.geocontext()): # doctest: +SKIP
                image_stack.max(axis="images").visualize(str(group_name), m, colormap="turbo") # doctest: +SKIP

        """
        if self.scenes_graft is None:
            raise Exception(
                "This ImageStack cannot be grouped because "
                "it no longer has image metadata. This can happen "
                "when, e.g., an ImageStack is created from a mathematical "
                "operation "
            )

        encoded_grouping_func = encode_function(grouping_func)
        groups = groupby(self.scenes_graft, encoded_grouping_func)

        return ImageStackGroupBy(self, groups)

    def reduce(
        self, reducer: Callable[[np.ndarray], np.ndarray], axis: str = "images"
    ) -> Union[Mosaic, ImageStack]:
        """
        Perform a reduction over either images or bands. Note that this does not mutate self.

        Parameters
        ----------
        reducer: Callable[[np.ndarray], np.ndarray]
            Function to perform the reduction
        axis: str
            Axis over which to reduce, either "bands" or "images"

        Returns
        -------
        new_obj: Union[Mosaic, ImageStack]
            Reduced object, either a Mosaic if axis is "images" or an ImageStack
            if axis is "bands"
        """

        if axis == "bands":
            kwargs = {
                "scenes_graft": self.scenes_graft,
                "product_id": self.product_id,
            }
        else:
            kwargs = {}

        return reduction(self, reducer, axis, **kwargs)

    def visualize(*args, **kwargs):
        raise NotImplementedError(
            "ImageStacks cannot be visualized. You must reduce this to a Mosaic before calling visualize."
        )

    def serialize(self):
        """Serializes this object into a json representation"""

        return ImageStackSerializationModel(
            full_graft=dict(self),
            scenes_graft=self.scenes_graft,
            product_id=self.product_id,
            bands=self.bands,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
        ).json()

    @classmethod
    def deserialize(cls, data: str) -> ImageStack:
        """Deserializes into this object from json

        Parameters
        ----------
        data : str
            The json representation of the object state

        Returns
        -------
        ImageStack
            An instance of this object with the state stored in data
        """
        return cls(**ImageStackSerializationModel.from_json(data).dict())


def dot(
    a: Union[ImageStack, np.ndarray], b: Union[ImageStack, np.ndarray]
) -> ImageStack:
    """
    Specific implementation of dot for ImageStack objects. Either a's type or b's
    must be ImageStack, or a subclass of ImageStack. The other supported type is
    numpy.ndarray. This function assumes that ImageStack arguments are proxy
    objects referring to images by bands by pixel rows by pixel columns, and
    that the operation will be repeated for each band and pixel. In particular
    this does not support a ImageStack object that is matrix-per-pixel.
    The behavior of dot is as follows:

    1. If both arguments are ImageStacks, or subclasses thereof, return a ImageStack
    object with a single image containing the inner product along the images of
    the input ImageStacks. Note that this function cannot detect a dimension
    mismatch, e.g. dot applied two ImageStacks with differing numbers
    of images.

    2. If one argument is a ImageStack and the other argument is a matrix,
    matrix-vector (or vector matrix) multiplication is performed per pixel.
    Again, this function cannot check dimension agreement.

    3. If one argument is a ImageStack and the other argument is a vector,
    perform a dot product along the ImageStack images.

    Parameters
    ----------
    a: Union[ImageStack, np.ndarray]
        First operand
    b: Union[ImageStack, np.ndarray]
        Second operand

    Returns
    -------
    product: Union[ImageStack, Mosaic]
        Product of a and b.
    """
    if not (issubclass(type(a), ImageStack) or issubclass(type(b), ImageStack)):
        raise NotImplementedError(
            f"`image_stack.dot` not implemented for {type(a)}, {type(b)}"
        )

    if issubclass(type(a), ImageStack):
        if issubclass(type(b), ImageStack):
            # ImageStack times ImageStack, multiply and sum along images,
            # and return a Mosaic
            return Mosaic(
                _apply_binary(
                    a,
                    b,
                    lambda aa, bb: np.einsum("ibrc,ibrc->brc", aa, bb),
                    lambda pa, pb, **kwargs: {},
                )
            )
        else:
            if not isinstance(b, np.ndarray):
                raise NotImplementedError(
                    f"`image_stack.dot` not implemented for {type(a)}, {type(b)}"
                )

            # ImageStack times numpy array
            if len(b.shape) == 2:
                # ImageStack time matrix -- perform the matrix multiplication
                # along the images.
                return ImageStack(
                    _apply_binary(
                        a,
                        as_compute_map(b),
                        lambda aa, bb: np.einsum("ibrc,ij->jbrc", aa, bb),
                        lambda pa, pb, **kwargs: {},
                    )
                )
            elif len(b.shape) == 1:
                # ImageStack time vector -- perform a dot product along the
                # image band, return a Mosaic.
                return Mosaic(
                    _apply_binary(
                        a,
                        as_compute_map(b),
                        lambda aa, bb: np.einsum("ibrc,i->brc", aa, bb),
                    )
                )
            else:
                raise Exception(
                    f'Incompatible dimension for "b" {b.shape} in image_stack.dot'
                )
    else:
        if not isinstance(a, np.ndarray):
            raise NotImplementedError(
                f"`image_stack.dot` not implemented for {type(a)}, {type(b)}"
            )

        # numpy array times ImageStack
        if len(a.shape) == 2:
            # Matrix times ImageStack -- perform the matrix multiplication
            # along the bands.
            return ImageStack(
                _apply_binary(
                    as_compute_map(a),
                    b,
                    lambda aa, bb: np.einsum("ij,jbrc->ibrc", aa, bb),
                    lambda pa, pb, **kwargs: {},
                )
            )
        elif len(a.shape) == 1:
            # ImageStack time vector -- perform a dot product along the image
            # band, return a Mosaic.
            return Mosaic(
                _apply_binary(
                    as_compute_map(a),
                    b,
                    lambda aa, bb: np.einsum("i,ibrc->brc", aa, bb),
                )
            )
        else:
            raise Exception(
                f'Incompatible dimension for "a" {a.shape} in `image_stack.dot`'
            )
