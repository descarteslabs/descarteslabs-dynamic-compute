from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from copy import deepcopy
from numbers import Number
from typing import Dict, List, Optional, Tuple, Union

import ipyleaflet
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
from .datetime_utils import normalize_datetime_or_none
from .dl_utils import get_product_or_fail
from .interactive.tile_url import validate_scales
from .operations import (
    _apply_binary,
    _clip_data,
    _concat_bands,
    _pick_bands,
    _rename_bands,
    adaptive_mask,
    create_mosaic,
    format_bands,
    is_op,
    masked_einsum,
    op_args,
    op_type,
    reset_graft,
    set_cache_id,
)
from .reductions import reduction
from .serialization import BaseSerializationModel


@dataclasses.dataclass
class MosaicSerializationModel(BaseSerializationModel):
    """State representation of a Mosaic instance"""

    graft: Dict
    product_id: str
    bands: Union[str, List[str]]
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None

    @classmethod
    def from_json(cls, data: str) -> MosaicSerializationModel:
        base_obj = super().from_json(data)
        base_obj.graft = reset_graft(base_obj.graft)
        return base_obj


class Mosaic(
    AddMixin,
    SubMixin,
    MulMixin,
    TrueDivMixin,
    FloorDivMixin,
    SignedMixin,
    ExpMixin,
    CompareMixin,
    NumpyReductionMixin,
    LogicalMixin,
    ComputeMap,  # Base class
):
    """
    Class wrapper around mosaic operations
    """

    _RETURN_PRECEDENCE = 1

    def __init__(
        self,
        graft: Dict,
        bands: Optional[Union[str, List[str]]] = None,
        product_id: Optional[str] = None,
        start_datetime: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        end_datetime: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    ):
        """
        Initialize a new instance of Mosaic. Users should rely on
        from_product_bands

        Parameters
        ----------
        graft: Dict
            Graft, which when evaluated will generate a bands x rows x cols array
        bands: Union[str, List[str]]
            Bands either as space separated names in a single string or a list
            of band names
        product_id: str
            Product id
        start_datetime: Optional[Union[str, datetime.date, datetime.datetime]]
            Optional initial cutoff
        end_datetime: Optional[Union[str, datetime.date, datetime.datetime]]
            Optional final cutoff
        """
        set_cache_id(graft)
        super().__init__(graft)
        self.bands = bands
        self.product_id = product_id
        self.start_datetime = normalize_datetime_or_none(start_datetime)
        self.end_datetime = normalize_datetime_or_none(end_datetime)

    def tile_layer(
        self,
        name=None,
        scales=None,
        colormap=None,
        checkerboard=True,
        log_level=logging.DEBUG,
        **parameter_overrides,
    ):
        """
        A `.DynamicComputeLayer` for this `Mosaic`.

        Generally, use `Mosaic.visualize` for displaying on map.
        Only use this method if you're managing your own ipyleaflet Map instances,
        and creating more custom visualizations.

        An empty  `Mosaic` will be rendered as a checkerboard (default) or blank tile.

        Parameters
        ----------
        name: str
            The name of the layer.
        scales: list of lists, default None
            The scaling to apply to each band in the `Mosaic`.

            If `Mosaic` contains 3 bands, ``scales`` must be a list like
            ``[(0, 1), (0, 1), (-1, 1)]``.

            If `Mosaic` contains 1 band, ``scales`` must be a list like ``[(0, 1)]``,
            or just ``(0, 1)`` for convenience

            If None, each 256x256 tile will be scaled independently.
            based on the min and max values of its data.
        colormap: str, default None
            The name of the colormap to apply to the `Mosaic`. Only valid if the
             `Mosaic` has a single band.
        checkerboard: bool, default True
            Whether to display a checkerboarded background for missing or masked data.
        log_level: int, default logging.DEBUG
            Only listen for log records at or above this log level during tile
            computation. See https://docs.python.org/3/library/logging.html#logging-levels
             for valid log levels.
        **parameter_overrides: JSON-serializable value, Proxytype, or ipywidgets.Widget
            Values---or ipywidgets---for any parameters that this `Mosaic` depends on.

            If this `Mosaic` depends on ``dc.widgets``, you don't have to pass anything
            for those---any widgets it depends on are automatically linked to the
            layer. However, you can override their current values (or widgets)
            by passing new values (or ipywidget instances) here.

            Values can be given as Proxytypes, or as Python objects like numbers,
            lists, and dicts that can be promoted to them.
            These arguments cannot depend on any parameters.

            If an ``ipywidgets.Widget`` is given, it's automatically linked, so
            updating the widget causes the argument value to change, and the
            layer to update.

            Once these initial argument values are set, they can be modified by
            assigning to `~.DynamicComputeLayer.parameters` on the returned
            `DynamicComputeLayer`.

            For more information, see the docstring to `ParameterSet`.

        Returns
        -------
        layer: `.DynamicComputeLayer`
        """
        from descarteslabs.dynamic_compute.interactive.layer import DynamicComputeLayer

        return DynamicComputeLayer(
            self,
            name=name,
            scales=scales,
            colormap=colormap,
            checkerboard=checkerboard,
            log_level=log_level,
            parameter_overrides=parameter_overrides,
        )

    @classmethod
    def from_product_bands(
        cls,
        product_id: str,
        bands: Union[str, List[str]],
        start_datetime: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        end_datetime: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        **kwargs,
    ) -> Mosaic:
        """
        Create a new Mosaic object

        Parameters
        ----------
        product_id: str
            ID of the product from which we want to access data
        bands: Union[str, List[str]]
            A space-separated list of bands within the product, or a list of strings.
        start_datetime: Optional[Union[str, datetime.date, datetime.datetime]
            Start date for mosaic
        end_datetime: Optional[Union[str, datetime.date, datetime.datetime]
            End date for mosaic


        Returns
        -------
        m: Mosaic
            New mosaic object.
        """

        _ = get_product_or_fail(product_id)
        start_datetime = normalize_datetime_or_none(start_datetime)
        end_datetime = normalize_datetime_or_none(end_datetime)
        bands = " ".join(format_bands(bands))

        graft = create_mosaic(product_id, bands, start_datetime, end_datetime, **kwargs)

        return cls(graft, bands, product_id, start_datetime, end_datetime)

    def pick_bands(self, bands: Union[str, List[str]]) -> Mosaic:
        """
        Create a new Mosaic object with the specified bands and
        the product-id of this Mosaic object

        Parameters
        ----------
        bands: str
            A space-separated list of bands within the product, or a list
            of bands as strings

        Returns
        -------
        m: Mosaic
            New mosaic object.
        """

        bands = format_bands(bands)

        return_key = self["returns"]
        return_value = self[return_key]

        if not (is_op(return_value) and op_type(return_value) == "mosaic"):
            return Mosaic(_pick_bands(self, json.dumps(format_bands(bands))))

        args = op_args(return_value)

        product_id = self[args[0]]
        original_bands = format_bands(self[args[1]])
        options = deepcopy(args[2])

        for key in options:
            options[key] = self[options[key]]
        options.pop("cache_id", None)

        if set(bands) > set(original_bands):
            raise Exception(
                f"selected bands {bands} are not a subset of the mosaic bands {original_bands}"
            )

        return Mosaic.from_product_bands(product_id, bands, **options)

    def rename_bands(self, bands):
        """Rename the bands of an array."""

        return Mosaic(_rename_bands(self, json.dumps(format_bands(bands))))

    def unpack_bands(
        self, bands: Union[str, List[str]]
    ) -> Union[Mosaic, Tuple[Mosaic, ...]]:
        """
        Create a tuple of new Mosaic objects with one for each bands.

        Parameters
        ----------
        bands: str
            A space-separated list of bands within the product.

        Returns
        -------
        m: Tuple[Mosaic, ...]
            New mosaic object per band passed in.
        """
        bands = format_bands(bands)
        if len(bands) > 1:
            return tuple([self.pick_bands([band]) for band in bands])
        else:
            return self.pick_bands([bands[0]])

    def mask(self, mask: ComputeMap) -> Mosaic:
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
        return Mosaic(
            _apply_binary(mask, self, adaptive_mask, lambda pa, pb, **kwargs: pb)
        )

    def concat_bands(self, other: Union[Mosaic, str, List[str]]) -> Mosaic:
        """
        Create a new Mosaic that stacks bands. This call does not
        mutate `this`

        Parameters
        ----------
        other: Union[Mosaic, str]
            concat the bands of this mosiac with those of other, if other is a Mosaic.
            otherwise assume other is a list of bands and concat this with those bands.

        Returns
        -------
        stack: Mosaic
            Mosaic object with stacked bands
        """

        if not isinstance(other, Mosaic):
            other = self.pick_bands(other)

        return Mosaic(_concat_bands(self, other))

    def clip(self, lo: Number, hi: Number) -> Mosaic:
        """
        Generate a new Mosaic that is bounded by low and hi.

        Parameters
        ----------
        lo: Number
            Lower bound
        hi: Number
            Upper bound

        Returns
        -------
        bounded: Mosaic
            New Mosaic object that is bounded
        """
        if not lo < hi:
            raise Exception(f"Lower bound ({lo}) is not less than upper bound ({hi})")

        return Mosaic(_clip_data(self, lo, hi))

    def reduce(self, reducer: str, axis: str = "bands"):
        """
        Call a reduction function on this Mosaic

        Args:
            reducer (Callable): function to reduce Mosaic
            axis (str): Axis over which to call the reducer, must be in ["bands"].

        Raises:
            NotImplementedError: axis must be `bands`

        Returns:
            Mosaic
        """
        if axis != "bands":
            raise NotImplementedError(
                f"Reduction over {axis} not implemented for Mosaic"
            )
        return reduction(self, reducer, axis)

    def visualize(
        self,
        name: str,
        map: ipyleaflet.leaflet.Map,
        colormap: Optional[str] = None,
        scales: Optional[List[List]] = None,
        checkerboard=True,
        **parameter_overrides,
    ) -> ipyleaflet.leaflet.TileLayer:
        """
        Visualize this Mosaic instance on a map. This call does not
        mutate `this`
        Parameters
        ----------
        name: str
            Name of this layer on the map
        map: ipyleaflet.leaflet.Map
            IPyleaflet map on which to add this mosaic as a layer
        colormap: str
            Optional colormap to use
        scales: list
            List of lists where each sub-list is a lower and upper bound. There must be
            as many sub-lists as bands in the mosaic

        Returns
        -------
        layer: lyr
            IPyleaflet tile layer on the map.
        """

        if scales is not None:
            scales = validate_scales(scales)
            if not isinstance(scales, list):
                raise Exception("Scales must be a list")
            for scale in scales:
                if len(scale) != 2:
                    raise Exception("Each entry in scales must have a min and max")

        for layer in map.layers:
            if layer.name == name:
                with layer.hold_url_updates():
                    layer.set_imagery(self, **parameter_overrides)
                    layer.set_scales(scales, new_colormap=colormap)
                    layer.checkerboard = checkerboard
                return layer
        else:
            layer = self.tile_layer(
                name=name,
                scales=scales,
                colormap=colormap,
                checkerboard=checkerboard,
                **parameter_overrides,
            )
            map.add_layer(layer)
            return layer

    def serialize(self):
        """Serializes this object into a json representation"""

        return MosaicSerializationModel(
            graft=dict(self),
            product_id=self.product_id,
            bands=self.bands,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
        ).json()

    @classmethod
    def deserialize(cls, data: str) -> Mosaic:
        """Deserializes into this object from json

        Parameters
        ----------
        data : str
            The json representation of the object state

        Returns
        -------
        Mosaic
            An instance of this object with the state stored in data
        """

        return cls(**MosaicSerializationModel.from_json(data).dict())


def property_propagation_for_dot(
    properties_a: dict, properties_b: dict, **kwargs
) -> dict:
    """
    Provide logic for property propagation used in the `dot` operation

    Parameters
    ----------
    properties_a: dict
        Properties for the first argument
    properties_b: dict
        Properties for the second argument

    Returns
    -------
    new_properties: dict
        Properties for the result of the dot operation.
    """
    new_properties = {}

    # Note that if a or b is a matrix it will have no padding.
    pad_a = properties_a.get("pad", None)
    pad_b = properties_b.get("pad", None)

    if pad_a and pad_b and pad_a != pad_b:
        raise Exception("Cannot dot objects with different padding")

    if pad_a:
        new_properties["pad"] = pad_a
    elif pad_b:
        new_properties["pad"] = pad_b

    pid_a = properties_a.get("product_id", None)
    pid_b = properties_b.get("product_id", None)

    if pid_a:
        new_properties["product_id"] = pid_a
        if pid_b and pid_a != pid_b:
            new_properties["other_product_id"] = pid_b
    elif pid_b:
        new_properties["product_id"] = pid_b

    return new_properties


def dot(a: Union[Mosaic, np.ndarray], b: Union[Mosaic, np.ndarray]) -> Mosaic:
    """
    Specific implementation of dot for Mosaic objects. Either a's type or b's
    must be Mosaic, or a subclass of Mosaic. The other supported type is
    numpy.ndarray. This function assumes that Mosaic arguments are proxy objects
    referring to bands by pixel rows by pixel columns, and that the operation
    will be repeated for each pixel. In particular this does not support
    a Mosaic object that is matrix-per-pixel. The behavior of dot is as follows:

    1. If both arguments are Mosaics, or subclasses thereof, return a Mosaic object
    with a single band containing the inner product along the bands of the input
    Mosaics. Note that this function cannot detect a dimension mismatch, e.g.
    dot applied two mosaics with differing numbers of bands.

    2. If one argument is a Mosaic and the other argument is a matrix, matrix-vector
    (or vector matrix) multiplication is performed per pixel. Again, this
    function cannot check dimension agreement.

    3. If one argument is a Mosaic and the other argument is a vector, perform a
    dot product along the mosaic bands.

    Parameters
    ----------
    a: Union[Mosaic, np.ndarray]
        First operand
    b: Union[Mosaic, np.ndarray]
        Second operand

    Returns
    -------
    product: Mosaic
        Product of a and b.
    """
    if not (issubclass(type(a), Mosaic) or issubclass(type(b), Mosaic)):
        raise NotImplementedError(
            f"`mosaic.dot` not implemented for {type(a)}, {type(b)}"
        )

    if issubclass(type(a), Mosaic):
        if issubclass(type(b), Mosaic):
            # Mosaic times mosaic, multiply and sum along bands, then replace
            # the band dimension.
            return Mosaic(
                _apply_binary(
                    a,
                    b,
                    lambda aa, bb: masked_einsum("irc,irc->rc", aa, bb)[None, ...],
                    property_propagation_for_dot,
                )
            )
        else:
            if not isinstance(b, np.ndarray):
                raise NotImplementedError(
                    f"`mosaic.dot` not implemented for {type(a)}, {type(b)}"
                )

            # Mosaic times numpy array
            if len(b.shape) == 2:
                # Mosaic time matrix -- perform the matrix multiplication along
                # the bands.
                return Mosaic(
                    _apply_binary(
                        a,
                        as_compute_map(b),
                        lambda aa, bb: masked_einsum("irc,ij->jrc", aa, bb),
                        property_propagation_for_dot,
                    )
                )
            elif len(b.shape) == 1:
                # Mosaic time vector -- perform the matrix multiplication
                # along the bands.
                return Mosaic(
                    _apply_binary(
                        a,
                        as_compute_map(b),
                        lambda aa, bb: masked_einsum("irc,i->rc", aa, bb)[None, ...],
                    )
                )
            else:
                raise Exception(
                    f'Incompatible dimension for "b" {b.shape} in mosaic.dot'
                )
    else:
        if not isinstance(a, np.ndarray):
            raise NotImplementedError(
                f"`mosaic.dot` not implemented for {type(a)}, {type(b)}"
            )

        # numpy array times Mosaic
        if len(a.shape) == 2:
            # Matrix times mosaic -- perform the matrix multiplication along the bands.
            return Mosaic(
                _apply_binary(
                    as_compute_map(a),
                    b,
                    lambda aa, bb: masked_einsum("ij,jrc->irc", aa, bb),
                    property_propagation_for_dot,
                )
            )
        elif len(a.shape) == 1:
            # Vector times mosaic -- perform the matrix multiplication along the bands.
            return Mosaic(
                _apply_binary(
                    as_compute_map(a),
                    b,
                    lambda aa, bb: masked_einsum("i,irc->rc", aa, bb)[None, ...],
                )
            )
        else:
            raise Exception(f'Incompatible dimension for "a" {a.shape} in `mosaic.dot`')
