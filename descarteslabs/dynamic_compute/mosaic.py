from __future__ import annotations

import json
import logging
from numbers import Number
from typing import Callable, List, Optional, Tuple, Union

import ipyleaflet
import numpy as np

from .compute_map import (
    AddMixin,
    CompareMixin,
    ComputeMap,
    ExpMixin,
    FloorDivMixin,
    MulMixin,
    SignedMixin,
    SubMixin,
    TrueDivMixin,
    as_compute_map,
)
from .operations import (
    _apply_binary,
    _apply_unary,
    _concat_bands,
    _pick_bands,
    _rename_bands,
    create_mosaic,
    format_bands,
    set_cache_id,
)


def adaptive_mask(mask, data):
    """
    This function creates a mask for data assuming that the mask may
    need to be extended to cover more dimensions.  If `data` has more
    leading dimensions than `mask`, `mask` is extended along those leading
    dimensions.

    The use case is as follows: Assume we have raster data whose shape is
    (bands, pixel-rows, pixel-cols). Assume was also have a per-pixel mask
    whose shape is just (pixel-rows, pixel-cols). This function will extend the
    mask to be (bands, pixel-rows, pixel-cols) to match the shape of the data.

    Note if the trailing dimensions of `data` don't match the dimensions of `mask`,
    this will fail -- we  don't check that present dimensions agree, we only add
    missing dimensions.

    Parmaters
    ---------
    mask: numpy.ndarray
        Mask to apply
    data: numpy.ndarray
        Data to mask

    Returns
    -------
    md : numpy.ma.core.MaskedArray
        Masked array.
    """

    # Squeeze out the  band dimension if it's a singleton
    mask = np.squeeze(mask)

    leading_shape = data.shape[: -len(mask.shape)]
    full_mask = np.outer(np.ones(leading_shape), mask).reshape(data.shape)

    return np.ma.masked_where(full_mask, data)


class Mosaic(
    # Supported operations as mix-ins
    AddMixin,
    SubMixin,
    MulMixin,
    TrueDivMixin,
    FloorDivMixin,
    SignedMixin,
    ExpMixin,
    CompareMixin,
    # Note: ComputeMap has default comparison operators. We need to
    # list CompareMixin before ComputeMap for CompareMixin operations
    # to be used instead of the default ComputeMap mixins
    ComputeMap,  # Base class
):
    """
    Class wrapper around mosaic operations
    """

    def __init__(self, g):
        set_cache_id(g)
        super().__init__(g)

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
        A `.DynamicComputeLayer` for this `Image`.

        Generally, use `Image.visualize` for displaying on map.
        Only use this method if you're managing your own ipyleaflet Map instances,
        and creating more custom visualizations.

        An empty  `Image` will be rendered as a checkerboard (default) or blank tile.

        Parameters
        ----------
        name: str
            The name of the layer.
        scales: list of lists, default None
            The scaling to apply to each band in the `Image`.

            If `Image` contains 3 bands, ``scales`` must be a list like ``[(0, 1), (0, 1), (-1, 1)]``.

            If `Image` contains 1 band, ``scales`` must be a list like ``[(0, 1)]``,
            or just ``(0, 1)`` for convenience

            If None, each 256x256 tile will be scaled independently.
            based on the min and max values of its data.
        colormap: str, default None
            The name of the colormap to apply to the `Image`. Only valid if the `Image` has a single band.
        checkerboard: bool, default True
            Whether to display a checkerboarded background for missing or masked data.
        log_level: int, default logging.DEBUG
            Only listen for log records at or above this log level during tile computation.
            See https://docs.python.org/3/library/logging.html#logging-levels for valid
            log levels.
        **parameter_overrides: JSON-serializable value, Proxytype, or ipywidgets.Widget
            Values---or ipywidgets---for any parameters that this `Image` depends on.

            If this `Image` depends on ``wf.widgets``, you don't have to pass anything for those---any
            widgets it depends on are automatically linked to the layer. However, you can override
            their current values (or widgets) by passing new values (or ipywidget instances) here.

            Values can be given as Proxytypes, or as Python objects like numbers,
            lists, and dicts that can be promoted to them.
            These arguments cannot depend on any parameters.

            If an ``ipywidgets.Widget`` is given, it's automatically linked, so updating the widget causes
            the argument value to change, and the layer to update.

            Once these initial argument values are set, they can be modified by assigning to
            `~.WorkflowsLayer.parameters` on the returned `WorkflowsLayer`.

            For more information, see the docstring to `ParameterSet`.

        Returns
        -------
        layer: `.WorkflowsLayer`
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

    @staticmethod
    def from_product_bands(
        product_id: str, bands: Union[str, List[str]], **kwargs
    ) -> Mosaic:
        """
        Create a new Mosaic object

        Parameters
        ----------
        product_id: str
            ID of the product from which we want to access data
        bands: Union[str, List[str]]
            A space-separated list of bands within the product, or a list of strings.

        Returns
        -------
        m: Mosaic
            New mosaic object.
        """
        formatted_bands = " ".join(format_bands(bands))
        return Mosaic(create_mosaic(product_id, formatted_bands, **kwargs))

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

        # This forces us to check if the bands are legit, before we create the graft
        format_bands(bands)

        return Mosaic(
            _pick_bands(self, json.dumps(format_bands(bands))),
        )

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
            _apply_binary(mask, self, adaptive_mask, lambda pa, pb, **kwargs: pb),
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

        return Mosaic(
            _concat_bands(self, other),
        )

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

        return Mosaic(
            _apply_unary(self, lambda a: np.clip(a, lo, hi)),
        )

    def visualize(
        self,
        name: str,
        map: ipyleaflet.leaflet.Map,
        colormap: Optional[str] = None,
        scales: Optional[List[List]] = None,
        # scales: Optional[list] = None,
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


def band_reduction(
    obj: Mosaic, reducer: Callable[[np.ndarray], np.ndarray], axis: Optional[str] = None
) -> Mosaic:
    """
    Implement reducer over bands. This call does not
    mutate `this`

    Parameters
    ----------
    reducer: Callable[[np.ndarray], np.ndarray]
        Function to reduce along the first axis of the array.
    axis: str
        Axis over which to sum, must be in ["bands", "scenes"]. If `axis`
        is not "bands" we dispatch to the parent function.

    Returns
    -------
    reduction: Mosaic
        Mosaic object representing reduction.
    """

    if axis != "bands":
        raise NotImplementedError(f"Reduction over {axis} not implemented for Mosaic")

    def strip_bands(d):
        if "bands" in d:
            d.pop("bands")

        return d

    # We insert a new axis (None in the brackets) so that data retains
    # a shape that is (num-bands, num-rows, num-cols)
    return Mosaic(
        _apply_unary(
            obj, lambda a: reducer(a, axis=0)[None, ...], prop_func=strip_bands
        ),
    )


# Append band reducers to the Mosaic class as bound methoeds.
Mosaic.sum = lambda mosaic, axis: band_reduction(mosaic, np.sum, axis=axis)
Mosaic.min = lambda mosaic, axis: band_reduction(mosaic, np.min, axis=axis)
Mosaic.max = lambda mosaic, axis: band_reduction(mosaic, np.max, axis=axis)
Mosaic.median = lambda mosaic, axis: band_reduction(mosaic, np.median, axis=axis)
Mosaic.mean = lambda mosaic, axis: band_reduction(mosaic, np.mean, axis=axis)
Mosaic.argmax = lambda mosaic, axis: band_reduction(mosaic, np.argmax, axis=axis)


def dot(a: Union[Mosaic, np.ndarray], b: Union[Mosaic, np.ndarray]) -> Mosaic:
    """
    Specific implementation of dot for Mosaic objects. Either a's type or b's must be Mosaic,
    or a subclass of Mosaic. The other supported type is numpy.ndarray. This function assumes
    that Mosaic arguments are proxy objects referring to bands by pixel rows by pixel columns,
    and that the operation will be repeated for each pixel. In particular this does not support
    a Mosaic object that is matrix-per-pixel. The behavior of dot is as follows:

    1. If both arguments are Mosaics, or subclasses thereof, return a Mosaic object with a single
    band containing the inner product along the bands of the input Mosaics. Note that this
    function cannot detect a dimension mismatch, e.g. dot applied two mosaics with differing numbers
    of bands.

    2. If one argument is a Mosaic and the other argument is a matrix, matrix-vector (or vector matrix)
    multiplication is performed per pixel. Again, this function cannot check dimension agreement.

    3. If one argument is a Mosaic and the other argument is a vector, perform a dot product
    along the mosaic bands.

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
            # Mosaic times mosaic, multiply and sum along bands, then replace the band dimension.
            return Mosaic(
                _apply_binary(
                    a,
                    b,
                    lambda aa, bb: np.einsum("irc,irc->rc", aa, bb)[None, ...],
                    lambda pa, pb, **kwargs: {},
                )
            )
        else:
            # Mosaic times numpy array
            if len(b.shape) == 2:
                # Mosaic time matrix -- perform the matrix multiplication along the bands.
                return Mosaic(
                    _apply_binary(
                        a,
                        as_compute_map(b),
                        lambda aa, bb: np.einsum("irc,ij->jrc", aa, bb),
                        lambda pa, pb, **kwargs: {},
                    )
                )
            elif len(b.shape) == 1:
                # Mosaic time vector -- perform the matrix multiplication along the bands.
                return Mosaic(
                    _apply_binary(
                        a,
                        as_compute_map(b),
                        lambda aa, bb: np.einsum("irc,i->rc", aa, bb)[None, ...],
                    )
                )
            else:
                raise Exception(
                    f'Incompatible dimension for "b" {b.shape} in mosaic.dot'
                )
    else:
        # numpy array times Mosaic
        if len(a.shape) == 2:
            # Matrix times mosaic -- perform the matrix multiplication along the bands.
            return Mosaic(
                _apply_binary(
                    as_compute_map(a),
                    b,
                    lambda aa, bb: np.einsum("ij,jrc->irc", aa, bb),
                    lambda pa, pb, **kwargs: {},
                )
            )
        elif len(a.shape) == 1:
            # Vector times mosaic -- perform the matrix multiplication along the bands.
            return Mosaic(
                _apply_binary(
                    as_compute_map(a),
                    b,
                    lambda aa, bb: np.einsum("i,irc->rc", aa, bb)[None, ...],
                )
            )
        else:
            raise Exception(f'Incompatible dimension for "a" {a.shape} in `mosaic.dot`')
