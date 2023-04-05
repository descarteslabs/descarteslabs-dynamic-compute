from numbers import Number
from typing import Type, Union

import numpy as np

from .compute_map import ComputeMap
from .image_stack import ImageStack
from .image_stack import dot as image_stack_dot
from .mosaic import Mosaic
from .mosaic import dot as mosaic_dot


def type_resolver(t: Type) -> Type:
    # Identify t with an element of a limited set of parent classes,
    # if such a relevant one exists. Otherwise return t.
    #
    # This function is used to help identify which implementation of
    # `dot` should be used. For example one may have a subclass of
    # Mosaic, e.g. SpectralMosaic, but the implementation for dot
    # should be the one used with Mosaic.
    for tt in [ImageStack, Mosaic, ComputeMap, np.ndarray, Number]:
        if issubclass(t, tt):
            return tt

    return t


def simple_mul(a, b):
    # Default implementation for dot for simple cases.
    return a * b


# A mapping from argument types, i.e. (type(operand1), type(operand2))
# to implementations of dot.
#
# Special case code such as this feels like starting down a path that
# ends in having to rework a bunch of stuff to handle some unanticipated
# case. If we have anticipate a limited set of proxy objects, this is
# workable.
dispatch_dict = {
    (Mosaic, Number): simple_mul,
    (Number, Mosaic): simple_mul,
    (ImageStack, Number): simple_mul,
    (Number, ImageStack): simple_mul,
    (Mosaic, np.ndarray): mosaic_dot,
    (np.ndarray, Mosaic): mosaic_dot,
    (Mosaic, Mosaic): mosaic_dot,
    (ImageStack, np.ndarray): image_stack_dot,
    (np.ndarray, ImageStack): image_stack_dot,
    (ImageStack, ImageStack): image_stack_dot,
}


def dot(
    a: Union[Number, np.ndarray, ComputeMap, Mosaic],
    b: Union[Number, np.ndarray, ComputeMap, Mosaic],
) -> ComputeMap:
    """
    A function akin to numpy.dot applied to per-pixel values in proxy objects.

    Parameters
    ----------
    a: Union[Number, numpy.array, Mosaic]
        First operand
    b: Union[Number, numpy.array, Mosaic]
        Second operand

    Returns
    -------
    product: Mosaic
        Proxy object for the product of a and b
    """

    # Implementation for dot
    def fail(aa, bb):
        raise NotImplementedError(f"dot not implemented for {type(aa)}, {type(bb)}")

    type_tuple = (type_resolver(type(a)), type_resolver(type(b)))

    return dispatch_dict.get(type_tuple, fail)(a, b)
