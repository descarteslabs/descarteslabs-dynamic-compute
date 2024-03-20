from numbers import Number
from typing import Type, Union

import numpy as np

from .compute_map import ComputeMap
from .image_stack import ImageStack
from .mosaic import Mosaic
from .operations import _dot_op


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


# Pairs of operand types that can be handled with multiplication.
simple_supported_type_pairs = {
    (Mosaic, Number),
    (Number, Mosaic),
    (ImageStack, Number),
    (Number, ImageStack),
}

# Pairs of operand types that can be handled with einsum via
# the dot keyword argument.
complex_supported_type_pairs = {
    (Mosaic, np.ndarray),
    (np.ndarray, Mosaic),
    (Mosaic, Mosaic),
    (ImageStack, np.ndarray),
    (np.ndarray, ImageStack),
    (ImageStack, ImageStack),
}


def _return_type(
    a: Union[np.ndarray, Mosaic, ImageStack], b: Union[np.ndarray, Mosaic, ImageStack]
) -> Type:
    """
    Compute the return type based on the input arguments

    Parameters
    ----------
    a: Union[np.ndarray, Mosaic, ImageStack]
        First operand
    b: Union[np.ndarray, Mosaic, ImageStack]
        Second operand

    Returns
    -------
    type: Type
        Return type for the two operands
    """
    a_type, b_type = type_resolver(type(a)), type_resolver(type(b))

    if ImageStack not in [a_type, b_type]:
        # All operatins involving any Mosaic, return a mosaic.
        return Mosaic

    if a_type == b_type:
        # Dot for two ImageStacks means a dot product along scenes,
        # Resulting in a Mosaic.
        return Mosaic

    if (
        a_type == np.ndarray
        and len(a.shape) == 1
        or b_type == np.ndarray
        and len(b.shape) == 1
    ):
        # Dot for an ImageStack and a vector means a dot-product along scenes,
        # Resulting in a Mosaic
        return Mosaic

    # One operand is an ImageStack and the other is a numpy array representing a matrix.
    # The dot operations is matrix multiplication along the scenes axis, and returns a new
    # ImageStack whose scenes are a linear combination of the input scenes.
    return ImageStack


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

    type_pair = (type_resolver(type(a)), type_resolver(type(b)))

    if type_pair[0] == np.ndarray and len(a.shape) not in [1, 2]:
        raise Exception(
            "First operand is an array, and so it must be a matrix of a vector, but has shape {a.shape}"
        )

    if type_pair[1] == np.ndarray and len(b.shape) not in [1, 2]:
        raise Exception(
            "Second operand is an array, and so it must be a matrix of a vector, but has shape {b.shape}"
        )

    if type_pair in simple_supported_type_pairs:
        return a * b

    if type_pair in complex_supported_type_pairs:
        a_type = type_pair[0].__name__
        b_type = type_pair[1].__name__

        return _return_type(a, b)(_dot_op(a, b, a_type, b_type))

    raise NotImplementedError(f"dot not implemented for {type(a)}, {type(b)}")
