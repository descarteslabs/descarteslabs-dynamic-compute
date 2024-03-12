from typing import Optional

from .compute_map import ComputeMap
from .operations import _reduction_op

BUILT_IN_REDUCERS = ["max", "min", "mean", "median", "sum", "std"]


def _get_return_type(axis, obj_type_str):
    from .image_stack import ImageStack
    from .mosaic import Mosaic

    if axis == "bands":
        if obj_type_str == "ImageStack":
            return ImageStack

        elif obj_type_str == "Mosaic":
            return Mosaic

    elif axis == "pixels":
        return Mosaic

    else:  # axis in (images)
        return Mosaic


def reduction(
    obj: ComputeMap,
    reducer: str,
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

    obj_type_str = type(obj).__name__

    return_type = _get_return_type(axis, obj_type_str)

    return return_type(_reduction_op(reducer, axis, obj_type_str, obj))
