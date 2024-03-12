from __future__ import annotations

import dataclasses
import datetime
import json
from copy import deepcopy
from numbers import Number
from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union

import descarteslabs as dl
import numpy as np

from .compute_map import (
    AddMixin,
    CompareMixin,
    ComputeMap,
    ExpMixin,
    FloorDivMixin,
    LogicalMixin,
    MulMixin,
    NumpyReductionMixin,
    SignedMixin,
    SubMixin,
    TrueDivMixin,
    as_compute_map,
)
from .datetime_utils import normalize_datetime, normalize_datetime_or_none
from .dl_utils import get_product_or_fail
from .mosaic import Mosaic
from .operations import (
    _apply_binary,
    _clip_data,
    _concat_bands,
    _index,
    _length,
    _pick_bands,
    _rename_bands,
    adaptive_mask,
    encode_function,
    filter_data,
    format_bands,
    is_op,
    masked_einsum,
    op_args,
    op_type,
    reset_graft,
    select_scenes,
    set_cache_id,
    stack_scenes,
)
from .reductions import reduction
from .serialization import BaseSerializationModel

AXIS_NAME_TO_INDEX_MAP = {"images": (0,), "bands": (1,), "pixels": (2, 3)}


def _optimize_image_stack_graft(
    graft: dict, bands: Union[str, List[str]]
) -> Optional[ImageStack]:
    """
    Given a graft for an image stack, determine if there is a more efficient ImageStack
    that evaluates to the samre output.

    A common use case is
    >>> image_stack = ImageStack.from_product_bands(<product_id>, <all bands>)
    >>> image_stack_for_viewing = image_stack.pick_bands("red green blue")

    If we evaluate `image_stack_for_viewing` we'll pull all the bands, and then throw out
    all but red, green, and blue. This is inefficient.

    This code is intended to support this case where pick_bands will call this function to
    create a simpler ImageStack if possible.

    Parameters
    ----------
    graft: dict
        Graft for an ImageStack.
    bands: List[str]
        List of bands that we actually use.

    Returns
    -------
    new_image_stack: Optional[ImageStack]
        Simpler ImageStack instance if possible, otherwise None.
    """

    bands = format_bands(bands)

    # Every graft has a "returns" key, so we start here.
    # key will act as a pointer into the graft as we walk
    # the graft.
    key = graft["returns"]

    filter_functions = []

    while True:

        if not is_op(graft[key]):
            return None

        args = op_args(graft[key])

        if op_type(graft[key]) in ["stack_scenes"]:
            # The first argument to the op is the source for the op.
            key = args[0]
            continue

        if op_type(graft[key]) in ["filter_data"]:
            # The first argument to the op is the source for the op.
            key = args[0]

            code_key = args[1]
            filter_functions.append(graft[code_key])
            continue

        if op_type(graft[key]) == "select_scenes":
            product_id_key = args[0]
            product_id = graft[product_id_key]
            bands_key = args[1]
            original_bands = format_bands(graft[bands_key])
            options = deepcopy(args[2])
            for options_key in options:
                options[options_key] = graft[options[options_key]]
            options.pop("cache_id", None)
            break

        # If we've reached this line of code, we've encountered
        # a key that doesn't correspond to stack_scenes, filter_scenes
        # of select_scenes, so this is not a "simple" image stack.
        return None

    # Make sure the new bands are a subset of the orignal bands.
    if set(bands) > set(original_bands):
        raise Exception(
            f"selected bands {bands} are not a subset of the mosaic bands {original_bands}"
        )

    bands = " ".join(bands)

    # Create a new ImageStack with the relevant bands.
    image_stack_graft = dict(
        ImageStack.from_product_bands(product_id, bands, **options)
    )

    for filter_function in filter_functions[::-1]:
        image_stack_graft = filter_data(image_stack_graft, filter_function)

    return ImageStack(image_stack_graft, bands=bands, **options)


@dataclasses.dataclass
class ImageStackSerializationModel(BaseSerializationModel):
    """State representation of a ImageStack instance"""

    full_graft: Dict
    product_id: str
    bands: Union[str, List[str]]
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None

    @classmethod
    def from_json(cls, data: str) -> ImageStackSerializationModel:
        base_obj = super().from_json(data)
        base_obj.full_graft = reset_graft(base_obj.full_graft)
        return base_obj


class ImageStack(
    AddMixin,
    SubMixin,
    MulMixin,
    TrueDivMixin,
    FloorDivMixin,
    LogicalMixin,
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
    # This allows us to compose filtering operations by creating a new "scenes"
    # graft that applies a new filtering operation to the existing "scenes" graft.
    #
    # The full graft uses the scenes graft to generate an ImageCollection and adds
    # instructions to accesss the raster data.

    _RETURN_PRECEDENCE = 2

    def __init__(
        self,
        full_graft: Dict,
        bands: Optional[Union[str, List[str]]] = None,
        product_id: Optional[str] = None,
        start_datetime: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        end_datetime: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        pad: Optional[int] = 0,
    ):
        """
        Initialize a new instance of ImageStack. Users should rely on
        from_product_bands

        Parameters
        ----------
        full_graft: Dict
            Graft, which when evaluated will generate a scenes x bands x rows
            x cols array
        bands: Union[str, List[str]]
            Bands either as space separated names in a single string or a list
            of band names
        product_id: str
            Product id
        start_datetime: Optional[Union[str, datetime.date, datetime.datetime]]
            Optional initial cutoff for an ImageStack
        end_datetime: Optional[Union[str, datetime.date, datetime.datetime]]
            Optional final cutoff for an ImageStack
        pad: Optional[int]
            Optional padding argument.
        """

        assert isinstance(pad, int)
        assert pad >= 0

        set_cache_id(full_graft)
        super().__init__(full_graft)
        self.bands = bands
        self.product_id = str(product_id)
        self.start_datetime = normalize_datetime_or_none(start_datetime)
        self.end_datetime = normalize_datetime_or_none(end_datetime)
        self.init_args = {
            "bands": self.bands,
            "product_id": self.product_id,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "pad": pad,
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
            product_id, formatted_bands, start_datetime, end_datetime, **kwargs
        )

        return cls(
            stack_scenes(scenes_graft, formatted_bands, **kwargs),
            formatted_bands,
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

        return ImageStack(filter_data(dict(self), encode_function(f)))

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

        bands = format_bands(bands)

        # See if there is an efficient way to do this.
        new_image_stack = _optimize_image_stack_graft(dict(self), bands)
        if new_image_stack:
            return new_image_stack

        return ImageStack(
            _pick_bands(self, json.dumps(bands)),
            bands=bands,
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

        if other.bands and self.bands:
            new_bands = self.bands + other.bands
        else:
            new_bands = None

        return ImageStack(_concat_bands(self, other), bands=new_bands)

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
            bands=self.bands,
            product_id=self.product_id,
        )

    def clip(self, lo: Number, hi: Number) -> ImageStack:
        """
        Generate a new ImageStack that is bounded by low and hi.

        Parameters
        ----------
        lo: Number
            Lower bound
        hi: Number
            Upper bound

        Returns
        -------
        bounded: ImageStack
            New ImageStack object that is bounded
        """
        if not lo < hi:
            raise Exception(f"Lower bound ({lo}) is not less than upper bound ({hi})")

        return ImageStack(_clip_data(self, lo, hi))

    def reduce(self, reducer: str, axis: str = "images") -> Union[Mosaic, ImageStack]:
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
            kwargs = {"product_id": self.product_id}
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


def dot_propagation_for_two_image_stacks(
    properties_a: List[dict], properties_b: List[dict]
) -> dict:
    """
    Handle property propagation for the case that the input to dot is two ImageStacks and
    the output is a Mosaic.

    Parameters
    ----------
    properties_a: List[dict]
        Per image properties of the first ImageStack
    properties_b: List[dict]
        Per image properties of the second ImageStack

    Returns
    -------
    properties: dict
        Properties for the resulting Mosaic
    """
    if len(properties_a) != len(properties_b):
        raise Exception(
            "Cannot apply dot to images stacks with different numbers of images"
        )

    pad_a = properties_a[0].get("pad", 0)
    pad_b = properties_b[0].get("pad", 0)

    if pad_a != pad_b:
        raise Exception("Cannot dot objects with different padding")

    properties = {"pad": pad_a}

    product_a = properties_a[0].get("product_id", None)
    product_b = properties_b[0].get("product_id", None)

    if product_a:
        properties["product_id"] = product_a

    if product_b and product_b != product_a:
        properties["other_product_ids"] = [product_b]

    bands_a = properties_a[0].get("bands", [])
    bands_b = properties_b[0].get("bands", [])

    if len(bands_a) and len(bands_b) and len(bands_a) != len(bands_b):
        raise Exception(
            "Cannot apply dot to ImageStacks with different numbers of bands"
        )

    if bands_a == bands_b:
        properties["bands"] = bands_a
    else:
        properties["bands"] = [
            f"{band_a}_dot_{band_b}" for band_a, band_b in zip(bands_a, bands_b)
        ]

    return properties


def keys_with_fixed_values(list_of_dict: List[Dict]) -> List[Hashable]:
    """
    Given a list of dictionaries return a list of keys that are
    present in all dictionaries and for which the value is the same
    for all dictionaries.

    Parameters
    ----------
    list_of_dicts: List[Dict]
        List of dictionaries for which we should find "stable" keys

    Returns
    -------
    stable_keys: List[Hashable]
        List of keys that were present in all dictionaries and had the same value.
    """
    stable_keys = []

    for key, value in list_of_dict[0].items():
        unique = True

        for dct in list_of_dict[1:]:
            try:
                if value != dct[key]:
                    unique = False
                    break
            except KeyError:
                unique = False
                break

        if unique:
            stable_keys.append(key)

    return stable_keys


def dot_property_propagation_for_image_stack_and_matrix(
    properties: List[dict], size: int
) -> List[dict]:
    """
    Handle property propagation for the case that the input to dot is an ImageStacks and
    a matrix.

    The matrix multiplication will result in a new image stack where each new image is a weighted sum
    of the input images. As such, the output images may not have an "acquired" entry that's meaninful.

    This function returns a properties list that contains entries that are unchanging in the
    input properties.


    Parameters
    ----------
    properties: List[dict]
        Per image properties of the ImageStack
    size: int
        Number of scenes in the output image stack

    Returns
    -------
    properties: List[dict]
        Properties for the resulting ImageStack
    """

    property_base = {
        key: properties[0][key] for key in keys_with_fixed_values(properties)
    }

    return [property_base for _ in range(size)]


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
                    lambda aa, bb: masked_einsum("ibrc,ibrc->brc", aa, bb),
                    dot_propagation_for_two_image_stacks,
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
                        lambda aa, bb: masked_einsum("ibrc,ij->jbrc", aa, bb),
                        lambda pa, pb, **kwargs: dot_property_propagation_for_image_stack_and_matrix(
                            pa, b.shape[1]
                        ),
                    )
                )
            elif len(b.shape) == 1:
                # ImageStack time vector -- perform a dot product along the
                # image band, return a Mosaic.
                return Mosaic(
                    _apply_binary(
                        a,
                        as_compute_map(b),
                        lambda aa, bb: masked_einsum("ibrc,i->brc", aa, bb),
                        lambda pa, pb, **kwargs: {"pad": pa[0].get("pad", 0)},
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
                    lambda aa, bb: masked_einsum("ij,jbrc->ibrc", aa, bb),
                    lambda pa, pb, **kwargs: dot_property_propagation_for_image_stack_and_matrix(
                        pb, a.shape[0]
                    ),
                )
            )
        elif len(a.shape) == 1:
            # ImageStack time vector -- perform a dot product along the image
            # band, return a Mosaic.
            return Mosaic(
                _apply_binary(
                    as_compute_map(a),
                    b,
                    lambda aa, bb: masked_einsum("i,ibrc->brc", aa, bb),
                    lambda pa, pb, **kwargs: {"pad": pb[0].get("pad", 0)},
                )
            )
        else:
            raise Exception(
                f'Incompatible dimension for "a" {a.shape} in `image_stack.dot`'
            )
