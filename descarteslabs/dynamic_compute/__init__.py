from importlib.metadata import version

__version__ = version("descarteslabs-dynamic-compute")

from .compute_map import (
    arccos,
    arcsin,
    arctan,
    as_compute_map,
    cos,
    e,
    log,
    log10,
    pi,
    resolution_x,
    resolution_y,
    sin,
    sqrt,
    tan,
)
from .dot import dot
from .groupby import ImageStackGroupBy
from .image_stack import ImageStack
from .interactive import map
from .mosaic import Mosaic
from .operations import gradient_x, gradient_y
from .proxies import Datetime, Proxytype, parameter

from . import catalog  # isort: skip

__all__ = [
    "as_compute_map",
    "resolution_x",
    "resolution_y",
    "catalog",
    "share_catalog_blob",
    "list_catalog_blobs",
    "delete_blob_in_catalog",
    "dot",
    "e",
    "pi",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "gradient_x",
    "gradient_y",
    "arccos",
    "arcsin",
    "arctan",
    "log",
    "log10",
    "Mosaic",
    "ImageStack",
    "ImageStackGroupBy",
    "map",
    "parameter",
    "Proxytype",
    "Datetime",
]
