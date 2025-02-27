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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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


def _get_pid_bands_pad(
    properties: Union[List[dict], dict]
) -> Tuple[Optional[str], Optional[List], Optional[int]]:
    """
    Given a properties object return the product id, bands and padding,
    if they are available

    Parameters
    ----------
    properties: Union[List[dict], dict]
        Properties for dynamic compute object, could be a dict, e.g. for a Mosaic,
        of a List, e.g. for an ImageStack.

    Returns
    -------
    pid: Optional[str]
        Product ID for this properties object if there is one, otherwise None
    bands: Optional[List]
        Bands associated with this properties object if data are available,
        otherwise None
    pad: Optional[int]
        Padding associated with this properties object if it's available,
        otherwise None
    """
    if isinstance(properties, List):
        if len(properties) > 0:
            return _get_pid_bands_pad(properties[0])
        else:
            return None, None, None

    return (
        properties.get("product_id", None),
        properties.get("bands", None),
        properties.get("pad", None),
    )


def _index(idx, arr, **kwargs):
    return graft_client.apply_graft(
        "index",
        arr,
        idx,
    )


def _length(image_stack, **kwargs):
    return graft_client.apply_graft(
        "length",
        image_stack,
    )


def _normalize_graft(graft: Dict, counter: Optional[Callable[[], int]] = None) -> Dict:
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
    counter: Optional[Callable[[], int]]
        Optional function to use to generate new keys

    Returns
    -------
    normalized_graft: Dict
        Graft with re-mapped keys.
    """

    sorted_non_return_keys = sorted(
        filter(lambda key: key != "returns", graft.keys()), key=lambda value: int(value)
    )

    if counter is None:
        key_mapping = {key: str(idx) for idx, key in enumerate(sorted_non_return_keys)}
    else:
        key_mapping = {key: str(counter()) for key in sorted_non_return_keys}

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


def reset_graft(graft: Dict) -> Dict:
    """
    Given a graft from a possibly different key-space, make sure
    update it to ensure that it doesn't collide with keys from
    the current key-space

    Parameters
    ----------
    graft: Dict
        Graft we would like to map into the current key space

    Returns
    -------
    remapped_graft: Dict
        Graft remapped into the current key-space
    """

    return _normalize_graft(graft, counter=graft_client.guid)


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
    classes: Optional[list] = None,
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
    if classes is not None:
        params["classes"] = classes
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
    resampler: dl.catalog.ResampleAlgorithm = dl.catalog.ResampleAlgorithm.NEAR,
) -> Dict:

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
    resampler : dl.catalog.ResampleAlgorithm
        default: dl.catalog.ResampleAlgorithm.NEAR. Optional resampling algorithm to use when mosaicking.
        dl.catalog.ResampleAlgorithm.BILINEAR option is also available.

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
        resampler=resampler,
    )


def _mask_op(data, mask, **kwargs):

    return graft_client.apply_graft(
        "mask",
        data,
        mask,
    )


def get_padding(graft):
    for i in list(graft.values()):
        if isinstance(i, list):
            if i[0] in ["select_scenes", "mosaic"]:
                pad_key = i[-1]["pad"]

    return graft[pad_key]


def _resolution_graft_x() -> dict:
    """
    Returns
    -------
    resolution_graft_x: dict
        Graft that evaluates to resolution_x.
    """
    return graft_client.apply_graft("resolution_x")


def _resolution_graft_y() -> dict:
    """
    Returns
    -------
    resolution_graft_y: dict
        Graft that evaluates to resolution_y.
    """
    return graft_client.apply_graft("resolution_y")


def _math_op(main_obj, operation, other_obj=None, **kwargs):
    """Apply a math operation between a graft and optionally another graft

    Params
    main_obj : graft
        The main graft that the operation is applying to
    operation : str
        The name of the operation, will be used by the backend to
        determine what operation to return
    other_obj : Union[graft, None]
        Optional other graft, defaults to None. Will be None for cases
        like abs(a), will be a valid graft for cases like a + b.

    Returns
    graft encoding the operation
    """

    from .image_stack import ImageStack
    from .mosaic import Mosaic

    if type(other_obj) in [ImageStack, Mosaic]:
        main_pad = get_padding(main_obj)
        other_pad = get_padding(other_obj)

        assert main_pad == other_pad, "Operands have different padding"

    return graft_client.apply_graft(
        "math",
        operation,
        main_obj,
        other_obj,
    )


def _reduction_op(reducer, axis, obj_type_str, obj, **kwargs):
    """Apply a reduction operation on graft. May be a built-in or custom operation

    Params
    reducer : str
        The reduction to apply to the graft
    axis : str
        The axis to apply the reduction to
    obj_type_str : str
        A string name of the object type
    obj : graft
        The graft that the reduction is applying to

    Returns
    graft encoding the reduction
    """

    return graft_client.apply_graft(
        "reduction",
        obj,
        reducer,
        axis,
        obj_type_str,
    )


def _dot_op(op1: dict, op2: dict, type1: str, type2: str) -> dict:
    """
    Create a graft for a dot operation

    Parameters
    ----------
    op1: dict
        Graft for the first operand
    op2: dict
        Graft for the second operand
    type1: str
        String of the type of the first operand
    type2: str
        String of the type of the second operand
    """
    return graft_client.apply_graft("dot", op1, op2, type1, type2)


def _func_op(obj, operation, **kwargs):
    """Apply a functional operation on a graft

    Params
    obj : graft
        The graft that the operation is applying to
    operation : str
        The name of the operation, will be used by the backend to
        determine what operation to return

    Returns
    graft encoding the operation
    """

    return graft_client.apply_graft(
        "functional",
        obj,
        operation,
    )


def _clip_data(obj, lo, hi, **kwargs):
    """Apply a math operation between a graft and optionally another graft

    Params
    obj : graft
        The graft that the clip is applying to
    lo : Union[int, float]
        The low value to clip data to
    hi : Union[int, float]
        The high value to clip data to

    Returns
    graft encoding the clipping
    """

    return graft_client.apply_graft("clip", obj, lo, hi)


def _fill_mask(obj, fill_val, **kwargs):
    """Apply a math operation between a graft and optionally another graft

    Params
    obj : graft
        The graft that the clip is applying to
    fill_val : Union[int, float, np.nan]
        The value to fill in a masked array

    Returns
    graft encoding the filling
    """

    return graft_client.apply_graft("filled", obj, fill_val)


def _band_op(main_obj, operation, bands=None, other_obj=None, **kwargs):
    """Apply a band operation between a graft and optionally another graft

    Params
    main_obj : graft
        The main graft that the operation is applying to
    operation : str
        The name of the operation, will be used by the backend to
        determine what operation to return
    other_obj : Union[graft, None]
        Optional other graft, defaults to None. Will be None for cases
        like abs(a), will be a valid graft for cases like a + b.

    Returns
    graft encoding the operation
    """

    from .image_stack import ImageStack
    from .mosaic import Mosaic

    if type(other_obj) in [ImageStack, Mosaic]:
        main_pad = get_padding(main_obj)
        other_pad = get_padding(other_obj)

        assert main_pad == other_pad, "Operands have different padding"

    return graft_client.apply_graft(
        "band_op",
        main_obj,
        operation,
        bands,
        other_obj,
    )


def select_scenes(
    product_id: str,
    bands: str,
    start_datetime: str,
    end_datetime: str,
    pad: int = 0,
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
        pad=pad,
    )


def stack_scenes(
    scenes_graft: Dict,
    bands: str,
    pad: int = 0,
    resampler: dl.catalog.ResampleAlgorithm = dl.catalog.ResampleAlgorithm.NEAR,
) -> Dict:
    """
    Given a graft that evaluates to an ImageCollection, create a graft that evaluates to
    an image stack array associated with that ImageCollection

    Parameters
    ----------
    scenes_graft: Dict
        A graft, which when evaluated generates an ImageCollection.
    bands: str
        Space separated list of band names
    pad: int
        Padding value, defaults to zero
    resampler : dl.catalog.ResampleAlgorithm
        default: dl.catalog.ResampleAlgorithm.NEAR. Optional resampling algorithm to use when mosaicking.
        dl.catalog.ResampleAlgorithm.BILINEAR option is also available.

    Returns
    -------
    stack_graft: Dict
        A graft, which when evaluated generates the ndarray associated with
        the ImageCollection

    """
    return graft_client.apply_graft(
        "stack_scenes", scenes_graft, bands, pad=pad, resampler=resampler
    )


def filter_by_id(stack_graft: Dict, id_list: str) -> Dict:
    """
    Apply filtering to an existing graft that evaluates to an ImageCollection

    Parameters
    ----------
    scenes_graft: Dict
        Graft, which when evaluated results in an ImageCollection object, e.g., a graft
        generated by select_scenes or this function.
    id_list: str
        json encoded list of Image IDs

    Returns
    -------
    filtered_graft: Dict
        Graft, which when evaluated results in an ImageCollection object containing images
        for which the filter function evaluates to true.
    """
    return graft_client.apply_graft("filter_by_id", stack_graft, id_list)


def filter_data(stack_graft: Dict, encoded_filter_func: str) -> Dict:
    """
    Apply filtering to an existing graft that evaluates to an ImageCollection

    Parameters
    ----------
    scenes_graft: Dict
        Graft, which when evaluated results in an ImageCollection object, e.g., a graft
        generated by select_scenes or this function.
    encoded_filter_func: str
        Encoded predicate, either a json serialized dl.catalog.properties.OpExpression,
        or a base64 encoded function. The function must take a dl.catalog.Image as input
        and return a bool.

    Returns
    -------
    filtered_graft: Dict
        Graft, which when evaluated results in an ImageCollection object containing images
        for which the filter function evaluates to true.
    """
    return graft_client.apply_graft("filter_data", stack_graft, encoded_filter_func)


def gradient_x(graft: Dict):

    return graft_client.apply_graft("math", "gradient_x", graft, None)


def gradient_y(graft: Dict):

    return graft_client.apply_graft("math", "gradient_y", graft, None)


def groupby(scenes_graft: Dict, encoded_key_func: str):
    return graft_client.apply_graft("groupby_data", scenes_graft, encoded_key_func)


def compute_aoi(
    graft: Dict, aoi: dl.geo.AOI, layer_id: str = None, **kwargs
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
            "parameters": kwargs,
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
    graft: Dict,
    lat: float,
    lon: float,
    layer_id: Optional[str] = None,
    **kwargs,
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
    if kwargs.get("parameters"):
        kwargs = kwargs["parameters"]

    def _get_most_common_value(array) -> int:
        arr, counts = np.unique(array, return_counts=True)
        return int(arr[counts == counts.max()][0])

    aoi = _geocontext_from_latlon(lat, lon)
    value_array, _ = compute_aoi(graft, aoi, layer_id, **kwargs)
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


# Support for legacy front-ends
# can be removed when <1.0 is no longer supported


def _adaptive_mask(mask, data):
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

        masked_data = np.ma.masked_array(data, False)
        index = tuple(
            [np.newaxis for _ in range(len(data.shape) - len(mask.shape))] + [...]
        )
        masked_data.mask |= mask[index]
        return masked_data

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

        masked_data = np.ma.masked_array(data, False)
        index = tuple(
            [np.newaxis for _ in range(len(data.shape) - len(mask.shape))] + [...]
        )
        masked_data.mask |= mask[index]
        return masked_data

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

    if isinstance(data, np.ma.MaskedArray):
        masked_data = data
    else:
        masked_data = np.ma.masked_array(data, False)
    if mask.shape[1] == 1:

        # The computational intent of the following three lines of code is
        #
        # for i in range(masked_data.shape[1]):
        #     masked_data.mask[:, i, :, :] = mask[:, 0, :, :]
        #
        # The conventional wisdom is that iteration is bad. However,
        # I can only seem to get the desired broadcast behavior by swapping
        # the fist two axes, doing the assignment, and then swapping back.

        temp = np.moveaxis(masked_data, (0, 1, 2, 3), (1, 0, 2, 3))
        temp.mask |= np.moveaxis(mask, (0, 1, 2, 3), (1, 0, 2, 3))
        masked_data = np.moveaxis(temp, (0, 1, 2, 3), (1, 0, 2, 3))
    else:
        masked_data.mask |= mask

    return masked_data


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
    try:
        return _adaptive_mask(mask, data)
    except Exception as e:
        if str(e).find("'bitwise_or' not supported for the input types") >= 0:
            raise Exception(
                f"Encountered an error applying a mask of type {mask.dtype} to an array of type {data.dtype}"
            )
        else:
            raise e


def _default_property_propagation(
    props0: Union[List[dict], dict],
    props1: Union[List[dict], dict],
    band_op: Optional[str] = "same",
    op_name: Optional[str] = None,
) -> Union[List[Dict], Dict]:
    """This function provides a default implementation of property propagation.

    Dynamic Compute tracks data and metadata through steps of an
    evaluation.  Binary operations take two pieces of data and two
    pieces of metadata (often called properties) to generate a new
    piece of data and a new piece of metadata. Dynamic Compute creates
    binary operations with `_apply_binary`.

    `_apply_binary` requires two operations, first, a function that
    combines the two pieces of input data, second a function that
    combines the two pieces of metadata to generate the metadata for
    the result of the binary operation.

    This function provides a default implementation of the second
    function required by `_apply_binary`. It may not be appropriate in
    all cases.

    Parameters
    ----------
    props0: Union[List[dict], dict]
        Properties (metadata for the first operand)
    props1: Union[List[dict], dict]
        Properties (metadata for the second operand)
    band_op: Optional[str] = "same"
        Either "same" meaning that bands are to be operated on together or
        "concat" meaning that input bands will be concatenated together.
    op_name: Optional[str] = None
        The name of the operation. This is used to create band names for the
        result

    Returns
    -------
    new_props: Union[List[dict], dict]
        Properties for the result of the binary operation
    """
    assert band_op in ["same", "concat"], f"Unrecognized band_op {band_op}"

    pid0, bands0, pad0 = _get_pid_bands_pad(props0)
    pid1, bands1, pad1 = _get_pid_bands_pad(props1)

    if pad0 and pad1 and pad0 != pad1:
        raise Exception(f"Operands have different padding {pad0} {pad1}")

    new_pad = None
    if pad0:
        new_pad = pad0
    elif pad1:
        new_pad = pad1
    else:
        new_pad = pad0

    new_bands = None

    if band_op == "same":
        # We aren't concatenating bands.

        if bands0 and bands1:
            # We have band information.

            if len(bands0) > 1 and len(bands1) > 1:
                # If either sets of bands has length 1, then we might be
                # using one to mask the other.

                if len(bands0) != len(bands1):
                    # We aren't masking and so we need the same number of bands.
                    raise Exception(
                        f"Operands have different numbers of bands {len(bands0)} {len(bands1)}"
                    )

                elif bands0 == bands1:
                    new_bands = bands0
                else:
                    new_bands = [
                        f"{band0}_{op_name}_{band1}"
                        for band0, band1 in zip(bands0, bands1)
                    ]

            else:
                if len(bands0) == 1:
                    new_bands = bands1
                else:
                    new_bands = bands0

        elif bands0:
            new_bands = bands0

        elif bands1:
            new_bands = bands1
    else:
        # We are concatenating bands

        if not bands0 or not bands1:
            raise Exception("Cannot concat bands when bands for one operand is missing")
        else:
            new_bands = bands0 + bands1

    new_pid = None
    other_pid = None
    if pid0 and not pid1:
        new_pid = pid0
    elif pid1 and not pid0:
        new_pid = pid1
    elif pid0 and pid1 and pid0 != pid1:
        new_pid = pid0
        other_pid = pid1

    if isinstance(props0, dict) and isinstance(props1, list):
        props0, props1 = props1, props0

    props0 = deepcopy(props0)

    if isinstance(props0, dict):

        props0.pop("shape", None)

        props0.pop("pad", None)
        if new_pad:
            props0["pad"] = new_pad
        else:
            props0["pad"] = 0

        props0.pop("bands", None)
        if new_bands:
            props0["bands"] = new_bands

        props0.pop("product_id", None)
        if new_pid:
            props0["product_id"] = new_pid

        other_product_ids = props0.get("other_product_ids", [])
        if other_pid:
            other_product_ids.append(other_pid)
            props0["other_product_ids"] = other_product_ids

        return props0

    elif isinstance(props0, list):

        for prop in props0:

            prop.pop("shape", None)

            prop.pop("pad", None)
            if new_pad:
                prop["pad"] = new_pad
            else:
                prop["pad"] = 0

            prop.pop("bands", None)
            if new_bands:
                prop["bands"] = new_bands

            prop.pop("product_id", None)
            if new_pid:
                prop["product_id"] = new_pid

            prop.pop("other_product_id", None)
            if other_pid:
                prop["other_product_id"] = other_pid

        return props0

    return {}


def _nan_mask(op: Union[np.ndarray, np.ma.MaskedArray]) -> np.ndarray:
    """
    Create an array where masked values are nans and non-masked values are zero
    Parameters
    ----------
    op: Union[np.ndarray, np.ma.MaskedArray]
        Operand for which we want the nan-mask
    Returns
    -------
    nan_mask: np.ndarray
        Nan-mask for operand
    """
    mask = np.zeros(op.shape)

    if not isinstance(op, np.ma.MaskedArray):
        return mask

    mask[op.mask] = np.nan

    return mask


def masked_einsum(
    signature: str,
    op1: Union[np.ndarray, np.ma.MaskedArray],
    op2: Union[np.ndarray, np.ma.MaskedArray],
) -> Union[np.ndarray, np.ma.MaskedArray]:
    """
    Compute an einsum that respects masks.
    Parameters
    ----------
    signature: str
        `np.einsum` signature
    op1: Union[np.ndarray, np.ma.MaskedArray]
        First operand
    op2: Union[np.ndarray, np.ma.MaskedArray]
        Second operand
    Returns
    -------
    product: Union[np.ndarray, np.ma.MaskedArray]
        Masked result for einsum
    """
    unmasked_result = np.einsum(signature, op1, op2)

    if not isinstance(op1, np.ma.MaskedArray) and not isinstance(
        op2, np.ma.MaskedArray
    ):
        return unmasked_result

    # Implementation idea: einsum is akin to matrix multiplicataion in that
    # it uses multiplication and addition to get a result. Masks are booleans
    # and for booleans multiplication is "and" and addition is "or", applying
    # einsum directly to the masks may not give the desired result.
    #
    # For each operatnd, we create a new mask where False (not masked) is 0 and
    # True (masked) is nan, and then take advantage that nans propagate as desired,
    # e.g.
    #
    #     nan + anything == anything + nan == nan
    #     nan * anything == anything * nan == nan
    #
    # We apply einsum on the two nan-masks, then mask the result for any nan result

    nan_mask1 = _nan_mask(op1)
    nan_mask2 = _nan_mask(op2)

    mask = np.isnan(np.einsum(signature, nan_mask1, nan_mask2))

    return np.ma.masked_array(unmasked_result, mask)


def update_kwarg(graft: dict, node_type: str, kwarg: str, value: str) -> dict:
    """
    Update, or set, a particular keyword argument within a type of graft node.

    Parameters
    ----------
    graft: dict
        Graft to be updated.
    node_type: str
        The type of node to upate, e.g. "mosaic"
    kwarg: str
        The keyword argument to change.
    value: str
        The value to set for the keyword argument

    Returns
    -------
    new_graft: dict
       A new graft with updated keyword argument values for the specified node_type
    """

    # Make a copy of the input graft, remove any cache ids, and ensure unique keys.
    # We don't want to modify the input in place.
    new_graft = reset_graft(graft_client.unset_all_cache_ids(deepcopy(graft)))

    # Create a new piece of graft with the desired value
    new_value_graft = graft_client.value_graft(value)
    new_value_key = new_value_graft.pop("returns")

    found_update = False

    # Walk through the new_graft.
    for key in new_graft:

        # Computational nodes, as opposed to value nodes, are always lists.
        if type(new_graft[key]) is not list:
            continue

        # Defensive coding, this should never happen.
        if len(new_graft[key]) == 0:
            continue

        # This is a computational node.
        node = new_graft[key]

        # Filter the node types we want.
        if node[0] != node_type:
            continue

        # Walk through the node looking for kwargs.
        node_has_kwargs = False
        for element in node:

            # The node's kwargs are a dict
            if type(element) is not dict:
                continue

            # NB Even if kwarg_value_key is not None, we don't want to
            # change the graft via
            #
            # new_graft[kwarg_value_key] = value
            #
            # It could be that other components of the graft, which we
            # want to stay unchanged, reference kwarg_value_key. For
            # this reason we create a new value entry and will
            # compress the graft when we're done.

            element.update({kwarg: new_value_key})
            node_has_kwargs = True

        # If the node did not have any kwargs, provide
        # this as one.
        if not node_has_kwargs:
            node.append({kwarg: new_value_key})

        found_update = True

    if found_update:
        new_graft.update(new_value_graft)

    return graft_client.compress_graft(new_graft)
