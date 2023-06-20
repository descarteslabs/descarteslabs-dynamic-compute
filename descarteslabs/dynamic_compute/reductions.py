from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from .compute_map import ComputeMap
from .operations import _apply_unary

IMAGESTACK_AXIS_NAME_TO_INDEX_MAP = {"images": (0,), "bands": (1,), "pixels": (2, 3)}
MOSAIC_AXIS_NAME_TO_INDEX_MAP = {"bands": 0}
AXIS_NAME_TO_INDEX_MAP = {
    "ImageStack": IMAGESTACK_AXIS_NAME_TO_INDEX_MAP,
    "Mosaic": MOSAIC_AXIS_NAME_TO_INDEX_MAP,
}


def reduction(
    obj: ComputeMap,
    reducer: Callable[[np.ndarray], np.ndarray],
    axis: Optional[str] = None,
    **kwargs,
) -> ComputeMap:
    """
    Perform a reduction over either images or bands. Note that this does not mutate obj.

    Parameters
    ----------
    obj: ImageStack or Mosaic
        ImageStack or Mosaic to be reduced
    reducer: Callable[[np.ndarray], np.ndarray]
        Function to perform the reduction
    axis: str
        Axis over which to reduce, either "bands", "images" or "pixels" for ImageStack.
        Only "bands" is supported for Mosaic.

    Returns
    -------
    new_obj: Union[Mosaic, ImageStack]
        Reduced object, either a Mosaic if axis is "images" or an ImageStack
        if axis is "bands"
    """
    from .image_stack import ImageStack
    from .mosaic import Mosaic

    obj_type_str = type(obj).__name__
    try:
        axis_from_name = AXIS_NAME_TO_INDEX_MAP[obj_type_str][axis]
    except KeyError:
        raise NotImplementedError(
            f"Reductions over {axis} not implemented for {obj_type_str}"
        )

    def strip_bands(props):
        props = deepcopy(props)
        if isinstance(props, list):
            for prop in props:
                if "bands" in prop:
                    prop.pop("bands")
        elif isinstance(props, dict):
            if "bands" in props:
                props.pop("bands")

        return props

    def mosaic_props(props):
        props = deepcopy(props)

        if isinstance(props, list):
            if not len(props):
                return {}
            props = props[0]

        return {
            "bands": props.get("bands", []),
            "pad": props.get("pad", 0),
            "product_id": props.get("product_id", ""),
        }

    if axis == "bands":
        if obj_type_str == "ImageStack":
            return ImageStack(
                _apply_unary(
                    obj,
                    lambda a: reducer(a, axis=axis_from_name)[:, None, ...],
                    prop_func=strip_bands,
                ),
                **kwargs,
            )
        elif obj_type_str == "Mosaic":
            return Mosaic(
                _apply_unary(
                    obj,
                    lambda a: reducer(a, axis=axis_from_name)[None, ...],
                    prop_func=strip_bands,
                )
            )
    elif axis == "pixels":
        # pass through properties from the image stack
        return Mosaic(_apply_unary(obj, lambda a: reducer(a, axis=axis_from_name)))
    else:  # axis in (images)
        return Mosaic(
            _apply_unary(
                obj, lambda a: reducer(a, axis=axis_from_name), prop_func=mosaic_props
            )
        )
