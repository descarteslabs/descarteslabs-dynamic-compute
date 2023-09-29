import base64
import copy
import functools
import hashlib
import io
import json
import os
import pickle
from copy import deepcopy
from importlib.metadata import version
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlencode

import cloudpickle
import descarteslabs as dl
import geojson  # type: ignore
import ipyleaflet  # type: ignore
import numpy as np
import requests

from .graft import client as graft_client
from .pyversions import PythonVersion

API_HOST = os.getenv(
    "API_HOST", "https://dynamic-compute.production.aws.descarteslabs.com"
)

SINGLE_POINT_BUFFER_VALUE = 0.0000001
WGS84_CRS = "EPSG:4326"
_python_major_minor_version = PythonVersion.from_sys().major_minor


class UnauthorizedUserError(requests.exceptions.HTTPError):
    """Raised when a user does not have the dynamic-compute-user group"""


def operation(func: Callable):
    """
    Decorator that defines a Python function as an operation that can be executed as part of a graft.
    """

    @functools.wraps(func)
    def wrapper_operation(*args, **kwargs):
        encoded_func = encode_function(func)
        graft = graft_client.apply_graft("code", encoded_func, *args, **kwargs)
        return graft

    return wrapper_operation


def encode_function(func: Callable) -> str:
    return base64.b64encode(cloudpickle.dumps(func)).decode("utf-8")


def format_bands(bands: Union[str, List[str]]) -> List[str]:
    """
    If input is a string of space separated tokens, convert it into a list of
    tokens. If input is a list, verify every element is a
    """

    if isinstance(bands, str):
        bands = bands.split(" ")
    elif not isinstance(bands, list):
        raise Exception("Bands must be a string or a list of strings")

    if not all(map(lambda x: isinstance(x, str), bands)):
        raise Exception("Bands must be a string or a list of strings")

    return bands


def _apply_unary(arg, value_func, prop_func=lambda x: x):
    @operation
    def encoded_func(a, args_props, *args, **kwargs):
        # Often property processing will detect and report and error better
        # than value processing. For this reason we process properties first.
        new_props = prop_func(args_props[0])
        new_value = value_func(a)
        return new_value, new_props

    return encoded_func(arg)


def _default_property_propagation(
    props0: Dict,
    props1: Dict,
    band_op: Optional[str] = "same",
    op_name: Optional[str] = None,
) -> Dict:
    """
    All graft operations, e.g. "code", "mosaic" and "array" return a value
    and properties associated with that value. For "mosaic" and "array" the
    returned properties are associated with underlying mosaic or array
    respectively. For "code" the returned properties will depend on operation
    and the input properties.

    This function provides a *default* implementation for binary operations.
    Note that there are plenty of binary operations where this is not the
    appropriate result.

    Note also that an essential part of this function is to raise a reasonable
    exception if the properties are incompatible.

    The logic is as follows: If there are multiple bands in both properties, the
    (ordered) bands lists have to be the same, i.e. ["red", "green", "blue"] is
    not the same as ["blue", "green", "red"]. Note this differs from WF, and
    likely will be changed.

    If one has one band, and the other has multiple bands, the multiple bands are
    returned.

    If one has bands and the other has a shape, the first entry in the shape must
    agree with the number of bands, and the bands are returned.

    If neither has bands, but both have shape, the shapes must agree and the shape
    is returned

    If one has shape and the other has neither bands nor shape, the shape is returned.

    If neither has shape or bands, the common agreed upon parameters are returend.

    Parameters
    ----------
    props0: Dict
        Properties associated with the first argument
    props1: Dict
        Properties associated with the second argument
    band_op: Optional[str]
        Band operation to perform, must be one of "same", "concat".
        "same" requires that if there are multiple bands in both properties,
        they must be the same. "concat" will generate new bands that is the
        concatination of the input bands.
    op_name: Optional[str]
        Name of operation, e.g. `sum` or `mul` to be used when creating a new
        band name out of two single bands, e.g. `red` + `vv` is `red_sum_vv`

    Returns
    -------
    props: Dict
        New resulting properties.
    """

    assert band_op in ["same", "concat"], f"Unrecognized band_op {band_op}"

    if isinstance(props0, dict):
        bands0 = props0.get("bands", [])
    else:
        bands0 = []

    if isinstance(props1, dict):
        bands1 = props1.get("bands", [])
    else:
        bands1 = []

    if band_op == "concat":
        if len(bands0) == 0 or len(bands1) == 0:
            raise Exception("Cannot concatenate bands for an object with no bands.")
        new_props = deepcopy(props0)
        new_props["bands"] = bands0 + bands1
        return new_props

    # Let's avoid some if clauses by ensuring that the first argument has at least as many
    # bands as the second.
    if len(bands1) > len(bands0):
        return _default_property_propagation(props1, props0, band_op=band_op)

    if len(bands1) > 1:
        if bands0 != bands1:
            raise Exception(f"Incompatible bands {bands0} and {bands1}")

        new_props = deepcopy(props0)

        if "product_id" in props1:
            product_id1 = props1["product_id"]
            if product_id1 != props0.get("product_id", ""):
                new_props.setdefault("other_product_ids", []).append(product_id1)

        return new_props

    if len(bands1) == 1 and len(bands0) > 1:
        new_props = deepcopy(props0)

        if "product_id" in props1:
            product_id1 = props1["product_id"]
            if product_id1 != props0.get("product_id", ""):
                new_props.setdefault("other_product_ids", []).append(product_id1)

        return new_props

    if len(bands1) == 1 and len(bands0) == 1:
        new_props = deepcopy(props0)

        new_props["bands"] = [f"{bands0[0]}_{op_name}_{bands1[0]}"]
        if "product_id" in props1:
            product_id1 = props1["product_id"]
            if product_id1 != props0.get("product_id", ""):
                new_props.setdefault("other_product_ids", []).append(product_id1)

        return new_props

    if len(bands1) == 0:
        return props0

    return {}


def _apply_binary(
    arg0: Dict,
    arg1: Dict,
    value_func: Callable[[Any, Any], Any],
    prop_func: Optional[
        Callable[[Dict, Dict, Optional[str], Optional[str]], Dict]
    ] = _default_property_propagation,
    band_op="same",
    op_name=None,
):
    """
    Create a graft that applies a binary operation to two input grafts.

    Parameters
    ----------
    args0: Dict
        Graft representing first operand
    args1: Dict
        Graft representing second operand
    value_func: Callable[[Any, Any], Any]
        Function for combining values to get a new value, must be cloudpickle-able. The arguments are
        value1, value2, the return type is the new value.
    prop_func: Optional[Callable[Dict, Dict, Optional[str], Optional[str]], Dict]
        Function for combining properties of the input operands. The arguments properties for value1,
        properties for value2, band_op (either "same" or "concat"), and op_name. This function handles
        how properties for binary operations should be handled. There are two important features of this
        First, we assume that how the properties are processed wont depend on how the values of the input.
        By and large this should be OK, as things like number of bands or array shape are encoded in
        properties. Second, this argument defaults to _default_property_propagation, which will cover a
        number of cases.
    band_op: Optional[str]
        How bands should be handled, defaults to "same", passed on to prop_func.
    op_name: Optional[str]
        Name of the operation to perform, defaults to None

    Returns
    -------
    encoded_func: Dict
        Encoded function applied to the input arguments, as a graft
    """

    @operation
    def encoded_func(a, b, args_props, *args, **kwargs):
        # Note that the order of the next two lines of code is important.
        # Either can raise an exception, the first will raise an exception
        # because of an incompatibility in the requested operation, e. g.
        # adding incompatible bands.
        #
        # If the first line of code raises an exception (and it were ignored)
        # the second *might* raise an exception. If it does likely the exception
        # will be less informative than the exception from the first line.
        #
        # For example adding a mosaic with "red green blue" to a mosaic with
        # "red green" will cause the first line to report incompatible bands and
        # list the bands, while the second line of code will likely raise an
        # exception like, "operands could not be broadcast together".
        #
        # Put another way, we test if the properties of the operands are compatible
        # before doing the value calculation.
        returned_properties = prop_func(
            args_props[0], args_props[1], band_op=band_op, op_name=op_name
        )
        returned_values = value_func(a, b)
        return returned_values, returned_properties

    return encoded_func(arg0, arg1)


@operation
def _pick_bands(arr, bands, args_props, **kwargs):
    bands = json.loads(bands)
    properties = deepcopy(args_props[0])

    if isinstance(properties, dict):
        # this is a Mosaic

        # Parse the bands to pick from JSON
        arr_bands = properties["bands"]

        # Get the indices that each picked band corresponds to in the array
        bands_idx = [arr_bands.index(band) for band in bands]

        # Pick the bands
        arr_bands = [arr[i] for i in bands_idx]
        arr_bands = np.ma.stack(arr_bands, axis=0)

        # Set the output bands according to the ones picked
        properties["bands"] = bands

    elif isinstance(properties, list):
        # this is an ImageStack

        if not properties:
            # There are no images in this image collection
            return arr, []

        arr_bands = properties[0]["bands"]

        # Get the indices that each picked band corresponds to in the array
        bands_idx = [arr_bands.index(band) for band in bands]

        # Pick the bands
        arr_bands = [arr[:, i] for i in bands_idx]
        arr_bands = np.ma.stack(arr_bands, axis=1)

        for prop in properties:
            # Set the output bands according to the ones picked
            prop["bands"] = bands

    return arr_bands, properties


@operation
def _rename_bands(arr, bands, args_props, **kwargs):
    # Parse the renamed bands from JSON
    bands = json.loads(bands)
    properties = deepcopy(args_props[0])

    def _rename(props, bands, arr_shape):
        # Rename the bands
        if "bands" in props.keys():
            _band_lists = []
            for _bands in [props["bands"], bands]:
                if isinstance(_bands, str):
                    _band_lists.append(_bands.split(" "))
                else:
                    _band_lists.append(_bands)

            assert len(_band_lists[0]) == len(
                _band_lists[1]
            ), "Mismatched bands in rename_bands"

            props["bands"] = bands
        else:
            props["bands"] = bands

            assert (
                len(bands) == arr_shape
            ), f"Mismatch between provided band names ({len(bands)}) and actual bands ({arr_shape})"
        return props

    if len(arr.shape) == 3:
        # this is a Mosaic
        properties = _rename(properties, bands, arr.shape[0])

    elif len(arr.shape) == 4:
        # this is an ImageStack
        for i, props in enumerate(properties):
            properties[i] = _rename(props, bands, arr.shape[1])

    return arr, properties


@operation
def _concat_bands(arr0, arr1, args_props, **kwargs):
    # Concatenate the bands

    properties0 = deepcopy(args_props[0])
    properties1 = deepcopy(args_props[1])

    def _concat_props(props0, props1, arr0, arr1):
        if "bands" in props0.keys():
            bands0 = props0["bands"]
        else:
            bands0 = [str(i) for i in range(len(arr0))]
        if "bands" in props1.keys():
            bands1 = props1["bands"]
        else:
            bands1 = [str(i + len(bands0)) for i in range(len(arr1))]

        props0["bands"] = bands0 + bands1

        if "product_id" in props1:
            if props1["product_id"] != props0.get("product_id", ""):
                props0.setdefault("other_product_ids", []).append(props1["product_id"])
        return props0

    if len(arr0.shape) == 3:
        # this is a Mosaic
        result = np.ma.concatenate((arr0, arr1), axis=0)
        properties0 = _concat_props(properties0, properties1, arr0, arr1)

    elif len(arr0.shape) == 4:
        # this is an ImageStack
        assert (
            arr0.shape[0] == arr1.shape[0]
        ), "Cannot concat bands, different number of images in each stack"

        result = np.ma.concatenate((arr0, arr1), axis=1)

        for prop0, prop1 in zip(properties0, properties1):
            prop0 = _concat_props(prop0, prop1, arr0, arr1)

    return result, properties0


def adaptive_mask(mask, data):
    """
    This function creates a mask for data assuming that the mask may
    need to be extended to cover more dimensions.  If `data` has more
    leading dimensions than `mask`, `mask` is extended along those leading
    dimensions.

    If `data` has more dimensions in the second position than `mask`, `mask`
    is extended along that dimensions.

    A few sample use cases follow: Assume we have raster data whose shape is
    (bands, pixel-rows, pixel-cols). Assume was also have a per-pixel mask
    whose shape is just (pixel-rows, pixel-cols). This function will extend the
    mask to be (bands, pixel-rows, pixel-cols) to match the shape of the data.

    Similarly, if the data is (scenes, bands, pixel-rows, pixel-cols), we extend
    the mask to (scenes, bands, pixel-rows, pixel-cols) to match. If instead, the
    mask is (scenes, bands=1, pixel-rows, pixel-cols) while the data is
    (scenes, bands=3, pixel-rows, pixel-cols), we will extend to match both
    cases.

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

    if mask.ndim not in [2, 3, 4]:
        raise Exception(
            (
                "Masks must be Mosaic of ImageStack objects. "
                f"Shape of input mask was {mask.shape}."
            )
        )

    if data.ndim not in [2, 3, 4]:
        raise Exception(
            (
                "Data to be masked must be Mosaic of ImageStack objects. "
                f"Shape of input data was {data.shape}."
            )
        )

    if (
        mask.ndim == 2
    ):  # this is a Mosiac with one band, that for whatever reason didn't come back as (1,row,cols)
        leading_shape = data.shape[: -len(mask.shape)]
        full_mask = np.outer(np.ones(leading_shape), mask).reshape(data.shape)

        return np.ma.masked_where(full_mask, data)

    if (
        mask.ndim == 3
    ):  # the mask is a Mosaic with multiple bands or a single band in an ImageStack
        if (
            data.ndim == 4
        ):  # data is an ImageStack, need to have the same number of bands
            if mask.shape[0] != data.shape[1] and mask.shape[0] != 1:
                raise Exception(
                    (
                        "Masks must be single-band or have the same number of bands as the ImageStack. "
                        f"Mask shape: {mask.shape}. ImageStack shape: {data.shape}."
                    )
                )
        else:  # data is a Mosaic, need to have the same number of bands
            if mask.shape[0] != data.shape[0] and mask.shape[0] != 1:
                raise Exception(
                    (
                        "Masks must be single-band or have the same number of bands as the Mosaic. "
                        f"Mask shape: {mask.shape}. Mosaic shape: {data.shape}."
                    )
                )

        if mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)
        leading_shape = data.shape[: -len(mask.shape)]
        full_mask = np.outer(np.ones(leading_shape), mask).reshape(data.shape)

        return np.ma.masked_where(full_mask, data)

    if mask.ndim == 4:  # the mask is an ImageStack
        if (
            data.ndim == 3
        ):  # the data is a Mosaic with multiple bands or a single band in an ImageStack
            raise Exception(
                "Cannot mask Mosaic with ImageStack, unless the ImageStack has been reduced over the 'images' axis."
            )
        else:
            if mask.shape[0] != data.shape[0]:
                raise Exception(
                    (
                        "ImageStack masks have same number of scenes as the ImageStack you are trying to mask. "
                        f"Mask shape: {mask.shape}. ImageStack shape: {data.shape}."
                    )
                )
            elif mask.shape[1] != data.shape[1] and mask.shape[1] != 1:
                raise Exception(
                    (
                        "ImageStack masks must be singular band or have the same number of "
                        "bands as the ImageStack you are trying to mask. "
                        f"Mask shape: {mask.shape}. ImageStack shape: {data.shape}."
                    )
                )

    if mask.shape[1] == 1:
        mask = np.hstack(data.shape[1] * [mask])

    return np.ma.masked_where(mask, data)


@operation
def _index(idx, arr, args_props, **kwargs):
    props = args_props[1]

    if not isinstance(props, list):
        raise Exception("Cannot index into a non-list")

    try:
        image_prop = props[idx]
    except IndexError:
        raise IndexError(
            f"Index {idx} is outside the bounds of of a list of size {props}"
        )

    return arr[idx], image_prop


@operation
def _length(arr, *args, **kwargs):
    return arr.shape[0], {"return_type": "int"}


def _normalize_graft(graft: Dict) -> Dict:
    """
    This function should only be used internally, in the context of generating cache keys.

    Given a graft, return a new graft where the non-return keys are sequentially ordered,
    starting with '0'.

    Note the purpose of this function is to aid in comparing grafts. The use case to consider is
    that the following:
    >>> a = Mosaic.from_product_bands(
    >>>     "sentinel-2", "red green blue", start_datetime="2021-01-01", end_datetime="2022-01-01"
    >>> )
    >>> b = Mosaic.from_product_bands(
    >>>     "sentinel-2", "red green blue", start_datetime="2021-01-01", end_datetime="2022-01-01"
    >>> )
    >>> dict(a) == dict(b)
    evaluates to False, because the keys in the grafts representing a and b are assigned
    sequentially, to avoid collision and enable composition of grafts. However, a and b are
    the same and we want a way to assess that.

    The important aspects of this function are:
    1. It helps enable comparison of grafts, by renaming keys in a normalized way.
    2. The return values of this function are grafts with common keys and should not be
       used to construct grafts that might be composed, since keys will likely collide.

    Parameters
    ----------
    graft: Dict
        Graft for which we want a representation with re-mapped keys

    Returns
    -------
    normalized_graft: Dict
        Graft with re-mapped keys.
    """

    sorted_non_return_keys = sorted(
        filter(lambda key: key != "returns", graft.keys()), key=lambda value: int(value)
    )

    key_mapping = {key: str(idx) for idx, key in enumerate(sorted_non_return_keys)}

    normalized_graft = {}

    for key in graft:

        new_value = copy.deepcopy(graft[key])

        if isinstance(new_value, list):
            for i, list_item in enumerate(new_value):

                if i == 0:
                    continue

                if isinstance(list_item, str):
                    new_value[i] = key_mapping.get(list_item, list_item)
                elif isinstance(list_item, dict):
                    for dict_key in list_item:
                        dict_value = list_item[dict_key]
                        list_item[dict_key] = key_mapping.get(dict_value, dict_value)
        elif isinstance(new_value, str):
            new_value = key_mapping.get(new_value, new_value)

        normalized_graft[key_mapping.get(key, key)] = new_value

    return normalized_graft


def set_cache_id(graft: Dict):
    """Set the cache ID of an operation.

    This is called when layers are created, as it is expected that these layers will be re-used in future operations.

    Parameters
    ----------
    graft : dict
        The graft where the cache ID will be set.
    """
    normalized_graft = _normalize_graft(graft)
    cache_id = hashlib.sha256(bytes(json.dumps(normalized_graft), "utf-8")).hexdigest()

    returned_key = graft["returns"]
    returned_op = graft[returned_key]
    op_kwargs = returned_op[-1] if isinstance(returned_op[-1], dict) else None

    if op_kwargs is None or "cache_id" not in op_kwargs:
        key = graft_client.client.guid()
        graft[key] = cache_id
    else:
        return

    if op_kwargs is not None:
        graft[returned_key][-1]["cache_id"] = key
    else:
        graft[returned_key].append({"cache_id": key})


def create_layer(
    name: str,
    graft: dict,
    colormap: Optional[str] = None,
    scales: Optional[list] = None,
    vector_tile_layer_styles: Optional[dict] = None,
    raster: bool = True,
):
    """Create an ipyleaflet raster or vector tile layer from a graft.

    Parameters
    ----------
    name : str
        Name of the layer.
    graft : dict
        The graft (ie. the directed acyclic graph) that describes how this tile layer should be formed.
    colormap : str, optional
        matplotlib colormap to apply to single band raster tiles.
    scales : list of list of float, optional
        Used to scale the intensities of the bands of a raster tile. Should be of the form [[0, 1]] for a single band
        raster layer and [[0, 1], [0, 1], [0, 1]] for a three band raster layer.
    vector_tile_layer_styles : dict, optional
        Styles to apply to the vector tile layer. See
        https://ipyleaflet.readthedocs.io/en/latest/layers/vector_tile.html for examples. Only used if this is a vector
        tile layer.
    raster : bool, default: True
        True if this layer is a raster tile layer. False if this layer is a vector tile layer.

    Returns
    -------
    lyr : ipyleaflet.TileLayer or ipyleaflet.VectorTileLayer
        Tile layer that can be added to an ipyleaflet map object.
    """
    # Create a layer from the graft
    response = requests.post(
        f"{API_HOST}/layers/",
        headers={"Authorization": dl.auth.Auth.get_default_auth().token},
        json={
            "graft": graft,
            "python_version": _python_major_minor_version,
            "dynamic_compute_version": version("descarteslabs-dynamic-compute"),
        },
        timeout=60,
    )

    try:
        response.raise_for_status()
    except Exception as e:
        if e.response.status_code == 403:
            raise UnauthorizedUserError(
                "User does not have access to dynamic-compute. "
                "If you believe this to be an error, contact support@descarteslabs.com"
            )
        else:
            raise e

    layer_id = json.loads(response.content.decode("utf-8"))["layer_id"]

    # URL encode query parameters
    params = {}
    params["python_version"] = _python_major_minor_version
    if scales is not None:
        params["scales"] = json.dumps(scales)
    if colormap is not None:
        params["colormap"] = colormap
    if vector_tile_layer_styles is None:
        vector_tile_layer_styles = {}
    query_params = urlencode(params)

    # Construct a URL to request tiles with
    url = f"{API_HOST}/layers/{layer_id}/tile/{{z}}/{{x}}/{{y}}?{query_params}"

    # Create an ipyleaflet raster or vector tile layer, as desired, and return it
    if raster:
        lyr = ipyleaflet.TileLayer(name=name, url=url, max_zoom=26, max_native_zoom=26)
    else:
        lyr = ipyleaflet.VectorTileLayer(
            name=name, url=url, vector_tile_layer_styles=vector_tile_layer_styles
        )
    return lyr


def create_mosaic(
    product_id: str,
    bands: str,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    pad: int = 0,
):
    """Mosaic a product in the Descartes Labs catalog.

    Parameters
    ----------
    product_id : str
        Catalog product ID of the product to mosaic.
    bands : str
        Space-delimited list of bands to mosaic.
    start_datetime : str, optional
        Datetime before which no scenes will be considered in the mosaicking operation.
    end_datetime : str, optional
        Datetime after which no scenes will be considered in the mosaicking operation.
    pad : int, default: 0
        Padding to apply to each tile, in pixels.

    Returns
    -------
    dict
        A graft who's result is the mosaiced catalog product.
    """

    return graft_client.apply_graft(
        "mosaic",
        product_id,
        bands,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        pad=pad,
    )


def select_scenes(
    product_id: str,
    bands: str,
    start_datetime: str,
    end_datetime: str,
) -> Dict:
    """
    Select, scenes based on date, from a product in the Descartes Labs catalog.

    Parameters
    ----------
    product_id : str
        Catalog product ID of the product to stack.
    bands : str
        Space-delimited list of bands to stack.
    start_datetime : str
        Date before which no scenes will be considered in the stacking operation.
    end_datetime : str
        Date after which no scenes will be considered in the stacking operation.

    Returns
    -------
    dict
        A graft whose evaluation is an ImageCollection object.
    """
    return graft_client.apply_graft(
        "select_scenes",
        product_id,
        bands,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )


def filter_scenes(scenes_graft: Dict, encoded_filter_func: str) -> Dict:
    """
    Apply filtering to an existing graft that evaluates to an ImageCollection

    Parameters
    ----------
    scenes_graft: Dict
        Graft, which when evaluated results in an ImageCollection object, e.g., a graft
        generated by select_scenes or this function.
    encoded_filter_func: str
        base64 encoded cloudpickled function. The function must take a dl.catalog.Image
        as input and return a bool.

    Returns
    -------
    filtered_graft: Dict
        Graft, which when evaluated results in an ImageCollection object containing images
        for which the filter function evaluates to true.
    """
    return graft_client.apply_graft("filter_scenes", scenes_graft, encoded_filter_func)


def stack_scenes(scenes_graft: Dict, bands: str) -> Dict:
    """
    Given a graft that evaluates to an ImageCollection, create a graft that evaluates to
    an image stack array associated with that ImageCollection

    Parameters
    ----------
    scenes_graft: Dict
        A graft, which when evaluated generates an ImageCollection.

    Returns
    -------
    stack_graft: Dict
        A graft, which when evaluated generates the ndarray associated with
        the ImageCollection

    """
    return graft_client.apply_graft("stack_scenes", scenes_graft, bands)


def filter_data(stack_graft: Dict, encoded_filter_func: str) -> Dict:
    """
    Apply filtering to an existing graft that evaluates to an ImageCollection

    Parameters
    ----------
    scenes_graft: Dict
        Graft, which when evaluated results in an ImageCollection object, e.g., a graft
        generated by select_scenes or this function.
    encoded_filter_func: str
        base64 encoded cloudpickled function. The function must take a dl.catalog.Image
        as input and return a bool.

    Returns
    -------
    filtered_graft: Dict
        Graft, which when evaluated results in an ImageCollection object containing images
        for which the filter function evaluates to true.
    """
    return graft_client.apply_graft("filter_data", stack_graft, encoded_filter_func)


def groupby(scenes_graft: Dict, encoded_key_func: str):
    return graft_client.apply_graft("groupby_data", scenes_graft, encoded_key_func)


def compute_aoi(
    graft: Dict, aoi: dl.geo.AOI, layer_id: str = None
) -> np.ma.MaskedArray:
    """Compute an AOI of a layer.

    Currently, only rasters are supported.

    Parameters
    ----------
    graft : dict
        The graft (ie. the directed acyclic graph) that describes how this tile layer should be formed.
    aoi : descarteslabs.geo.GeoContext
        GeoContext for which to compute evaluate this ComputeMap
    layer_id: Optional str
        layer id to reuse if supplied

    Returns
    -------
    arr : numpy.ma.MaskedArray
        The computed AOI.
    """

    import descarteslabs

    if isinstance(
        aoi,
        (
            descarteslabs.core.common.geo.geocontext.AOI,
            descarteslabs.core.common.geo.geocontext.DLTile,
            descarteslabs.core.common.geo.geocontext.XYZTile,
        ),
    ):
        aoi = dl.geo.AOI(
            geometry=aoi.geometry,
            resolution=aoi.resolution,
            crs=aoi.crs,
            align_pixels=aoi.align_pixels if hasattr(aoi, "align_pixels") else True,
            bounds=aoi.bounds,
            bounds_crs=aoi.bounds_crs,
            shape=aoi.shape if hasattr(aoi, "shape") else None,
            all_touched=aoi.all_touched,
        )
    else:
        raise TypeError(f"`compute` not implemented for AOIs of type {type(aoi)}")

    if not layer_id:
        # Create a layer from the graft if an id isn't supplied
        # NOTE: This is sort of redundant, but layer IDs are hashes so it won't
        # result in duplicates of existing layers
        response = requests.post(
            f"{API_HOST}/layers/",
            headers={"Authorization": dl.auth.Auth.get_default_auth().token},
            json={
                "graft": graft,
                "python_version": _python_major_minor_version,
                "dynamic_compute_version": version("descarteslabs-dynamic-compute"),
            },
            timeout=60,
        )

        try:
            response.raise_for_status()
        except Exception as e:
            if e.response.status_code == 403:
                raise UnauthorizedUserError(
                    "User does not have access to dynamic-compute. "
                    "If you believe this to be an error, contact support@descarteslabs.com"
                )
            else:
                raise e

        layer_id = json.loads(response.content.decode("utf-8"))["layer_id"]

    # Compute the AOI
    response = requests.post(
        f"{API_HOST}/layers/{layer_id}/aoi",
        headers={"Authorization": dl.auth.Auth.get_default_auth().token},
        json={
            "geometry": geojson.Feature(geometry=aoi.geometry)["geometry"],
            "resolution": aoi.resolution,
            "crs": aoi.crs,
            "align_pixels": aoi.align_pixels,
            "bounds": aoi.bounds,
            "bounds_crs": aoi.bounds_crs,
            "shape": aoi.shape,
            "all_touched": aoi.all_touched,
            "python_version": _python_major_minor_version,
            "dynamic_compute_version": version("descarteslabs-dynamic-compute"),
        },
    )

    try:
        response.raise_for_status()
    except Exception as e:
        if e.response.status_code == 403:
            raise UnauthorizedUserError(
                "User does not have access to dynamic-compute. "
                "If you believe this to be an error, contact support@descarteslabs.com"
            )
        else:
            raise e

    buf = io.BytesIO(response.content)
    payload = pickle.load(buf)
    return payload["array"], payload["properties"]


def value_at(
    graft: Dict, lat: float, lon: float, layer_id: Optional[str] = None
) -> List[float]:
    """
    Return the mean values for each band of a graft at a specific location

    Parameters
    ----------
    graft : dict
        The graft (ie. the directed acyclic graph) that describes how this tile layer should be formed.
    lat
        latitude of the point to evaluate
    lon
        longitude of the point to evaluate
    layer_id: Optional str
        layer id to reuse if supplied

    Returns
    -------
        list of numbers
    """

    def _get_most_common_value(array) -> int:
        arr, counts = np.unique(array, return_counts=True)
        return int(arr[counts == counts.max()][0])

    aoi = _geocontext_from_latlon(lat, lon)
    value_array, _ = compute_aoi(graft, aoi, layer_id)
    if len(value_array.shape) > 1:
        if np.issubdtype(value_array.dtype.type, np.bool_):
            # if we're dealing with booleans, return the most common value
            return list(map(_get_most_common_value, value_array))
        # otherwise, return each mean value per band
        return list(map(np.mean, value_array))
    return list(value_array)


def _geocontext_from_latlon(lat: float, lon: float) -> dl.geo.AOI:
    """
    Creates a tiny AOI from a lat/lon location. Private helper method for value_at, should only be called internally.

    Parameters
    ----------
    lat
        Latitude to create and center the AOI
    lon
        Longitude to create and center the AOI

    Returns
    -------
        dl.geo.AOI
    """
    from shapely.geometry import Point

    buffer = SINGLE_POINT_BUFFER_VALUE
    xy = Point(lon, lat)
    bounds = xy.buffer(buffer).bounds
    return dl.geo.AOI(
        bounds=bounds, crs=WGS84_CRS, shape=(1, 1), all_touched=True, align_pixels=True
    )


def is_op(graft_node: Any) -> bool:
    """
    Determine if a node in a graft is an operation.

    Parameters
    ----------
    graft_node: Any
        Node from a graft

    Returns
    -------
    is_op: bool
        True if the graft_node is an operation
    """

    return isinstance(graft_node, list)


def op_type(graft_op_node: list) -> str:
    """
    Determine the type of a graft operation

    Parameters
    ----------
    graft_op_node: list
        Graft node that is an operation

    Returns
    -------
    name: str
        Operation name
    """

    if not is_op(graft_op_node):
        raise ValueError(
            "Cannot determine op-type of graft node that is not an operation"
        )

    return graft_op_node[0]


def op_args(graft_op_node: list) -> list:
    """
    Determine the arguments of a graft operation

    Parameters
    ----------
    graft_op_node: list
        Graft node that is an operation

    Returns
    -------
    name: str
        Operation name
    """

    if not is_op(graft_op_node):
        raise ValueError("Cannot get op args of graft node that is not an operation")

    return graft_op_node[1:]
