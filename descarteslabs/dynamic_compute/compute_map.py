# This can be remoted with Python 3.11
# https://peps.python.org/pep-0673/
from __future__ import annotations

import json
import sys
from copy import copy
from io import StringIO
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Type, Union

import descarteslabs as dl
import numpy as np

from .graft.client import client as graft_client
from .graft.interpreter.interpreter import interpret
from .graft.syntax import syntax as graft_syntax
from .operations import _apply_binary, _apply_unary, compute_aoi


class DotDict(dict):
    """
    dot-notation access to top-level dictionary attributes

    From https://stackoverflow.com/a/23689767
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


type_map = {
    "int": int,
}


class Capturing(list):
    """
    A class for capturing stdout as a list of strings.
    Lifted from https://stackoverflow.com/a/16571630
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# Stub functions for primitives. Note we could just use
# a single no-op function, but the function name
# appears in the output
def code(*args, **kwargs):
    return None


def mosaic(*args, **kwargs):
    return None


def select_scenes(*args, **kwargs):
    return None


def filter_scenes(*args, **kwargs):
    return None


def stack_scenes(*args, **kwargs):
    return None


class ComputeMap(dict):
    """
    A wrapper class to support operations on grafts. Proxy objects should all be
    descended from ComputeMap
    """

    def __str__(self):
        obj_str = str(type(self))
        if obj_str.startswith("<class '"):
            return obj_str.split("'")[1]
        return obj_str

    def __repr__(self):

        with Capturing() as output:
            interpret(
                dict(self),
                builtins=[
                    ("code", code),
                    ("mosaic", mosaic),
                    ("select_scenes", select_scenes),
                    ("filter_scenes", filter_scenes),
                    ("stack_scenes", stack_scenes),
                ],
                debug=True,
            )()

        return "\n".join(output)

    def __init__(self, d):
        """
        Initialize a ComputeMap instance from a dictionary. If the
        dictionary is not a valid graft, raise a ValueError

        Parameters
        ----------
        d : dict
            Graft from which we intialize the compute map
        """

        if not graft_syntax.is_graft(d):
            raise ValueError("Invalid graft: " + json.dumps(d))

        super().__init__(d)
        self.return_val = "all"

    def __getattr__(self, attr):
        # Provide a way to evaluate to *just* the raster data or *just* the properties.
        # This is in support of a Workflows like interface.
        if attr not in ["properties", "ndarray"]:
            return super().__getattr__(self, attr)

        new_compute_map = copy(self)
        new_compute_map.return_val = attr
        return new_compute_map

    def compute(
        self, aoi: dl.geo.AOI
    ) -> Union[
        np.ma.MaskedArray,  # We're returning just data
        List,  # We're returning just properties, and they are a list
        Dict,  # We're returning just properties, and they are a dict
        DotDict,  # We're returning both data and properties as a DotDict
    ]:
        """
        Evaluate this ComputeMap for a particular AOI

        Parameters
        ----------
        aoi : descarteslabs.geo.AOI
            GeoContext for which to compute evaluate this ComputeMap

        Returns
        -------
        results : Union[Array, List, Dict, DotDict]
            Evaluation of self for this AOI. The return will be either
            an array, properties as a list, properties as dict, or both
            as a DotDict
        """
        value, properties = compute_aoi(self, aoi)

        if "return_type" in properties:
            value = type_map[properties["return_type"]](value)

        if self.return_val == "ndarray":
            return value

        if self.return_val == "properties":
            return properties

        return DotDict({"ndarray": value, "properties": properties})

    def to_imagery(self):
        # Compatibility with Workflows.
        new_compute_map = copy(self)
        new_compute_map.return_val = "all"
        return new_compute_map


def as_compute_map(a: Union[Number, Dict, ComputeMap, np.ndarray]) -> ComputeMap:
    """
    Return the input as a ComputeMap or raise an exception if this isn't possible.

    Parameters
    ----------
    a : Union[ComputeMap, Number, Dict, numpy.ndarray]
    Value to be represented as a ComputeMap

    Return
    ------
    cma : ComputeMap
    Input a as a ComputeMap instance.
    """

    if isinstance(a, Number) or isinstance(a, np.ndarray):
        return ComputeMap(graft_client.value_graft(a))

    if not isinstance(a, ComputeMap):
        # `a` is a Dict, but not a ComputeMap. If `a` it isn't a graft,
        # the ComputeMap constructor will throw the necessary exception.
        return ComputeMap(a)

    return a


def type_max(t1: Type, t2: Type) -> Type:
    """
    Return the more general of two types. If either t1 or t2 is a number or array return the other.
    If neither is a number or array and one is descended from the other return the ancestor type.
    If neither is a number or array and neither is descended from the other, raise a TypeError.

    This is used in a default implementation for computing the return types of binary
    operations. This may be insufficient for certain future classes, but can be overridden
    for classes in which this is the case.

    Parameters
    ----------
    t1 : Type
        First type
    t2 : Type
        Second type

    Returns
    -------
    t : Type
        More general of the types
    """
    if issubclass(t1, Number) or issubclass(t1, np.ndarray):
        return t2

    if issubclass(t2, Number) or issubclass(t2, np.ndarray):
        return t1

    if issubclass(t2, t1):
        return t1

    if issubclass(t1, t2):
        return t2

    raise TypeError(f"Types {t1} and {t2} are incompatible")


def binary_op(
    a: Union[Number, np.ndarray, ComputeMap],
    b: Union[Number, np.ndarray, ComputeMap],
    f: Callable[[Any, Any], Any],
    op_name: Optional[str] = None,
) -> ComputeMap:
    """
    Given two compute maps, create a new compute map by applying a binary operation to the two.

    The types a and b are assessed for compatibility -- one must be a number, or one must be a
    subclass of the other, or they must be the same class.

    Parameters
    ----------
    a: ComputeMap
        First operand
    b: ComputeMap
        Second operand
    f: Callable[[Any, Any], Any]
        Operation that combines the evalaution of a with the evaluation of b to produce a new value.
    op_name: Optional[str]
        Optional name for the operation

    Returns
    -------
    r: ComputeMap
        ComputeMap instance resulting from the operation f applied to the operands.
    """
    return_type = type_max(type(a), type(b))

    a = as_compute_map(a)
    b = as_compute_map(b)

    return_value = _apply_binary(a, b, f, op_name=op_name)

    return return_type(return_value)


#
# Mix-ins for a number of math operations
#


class AddMixin:
    def __add__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a + b, op_name="sum")

    def __radd__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a + b, op_name="sum")


class SubMixin:
    def __sub__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a - b, op_name="sub")

    def __rsub__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(other, self, lambda a, b: b - a, op_name="sub")


class MulMixin:
    def __mul__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a * b, op_name="mul")

    def __rmul__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a * b, op_name="mul")


class TrueDivMixin:
    def __truediv__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a / b, op_name="div")

    def __rtruediv__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: b / a, op_name="div")


class FloorDivMixin:
    def __floordiv__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a // b, op_name="floordiv")

    def __rfloordiv__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: b // a, op_name="floordiv")


class SignedMixin:
    def __abs__(self) -> ComputeMap:
        return type(self)(_apply_unary(self, lambda a: abs(a)))

    def __neg__(self) -> ComputeMap:
        return type(self)(_apply_unary(self, lambda a: -a))


class ExpMixin:
    def __pow__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a**b, op_name="exp")

    def __rpow__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: b**a, op_name="exp")


class CompareMixin:
    def __eq__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a == b, op_name="cmp")

    def __ne__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a != b, op_name="cmp")

    def __gt__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a > b, op_name="cmp")

    def __ge__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a >= b, op_name="cmp")

    def __lt__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a < b, op_name="cmp")

    def __le__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a <= b, op_name="cmp")


class LogicalMixin:
    def __and__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a & b, op_name="and")

    def __rand__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a & b, op_name="and")

    def __or__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a | b, op_name="or")

    def __ror__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(self, other, lambda a, b: a | b, op_name="or")


#
# generator for functional operations
#


def _functional_op(f):
    """
    Generate a function acting on grafts given a function acting on values

    Parameters
    ----------
    f: Callable[[Any], Any]
        Function acting on values

    Returns
    -------
    op: Callable[[Union[Number, Dict, ComputeMap]], ComputeMap]
        Function that takes a value that is, or can be made into graft, and returns
        a ComputeMap (graft) that applies f
    """

    def op(arg: Union[Number, Dict, ComputeMap]) -> ComputeMap:
        """
        Create a ComputeMap that applies f to the argument
        Parameters
        ----------
        arg : Union[Number, Dict, ComputeMap]
            Item for which we want to apply f
        Returns
        -------
        s : ComputeMap
            Representation of f applied to arg
        """
        if issubclass(type(arg), Number):
            return f(arg)

        return type(arg)(_apply_unary(as_compute_map(arg), f))

    return op


#
# functional operations
#

sqrt = _functional_op(np.sqrt)

cos = _functional_op(np.cos)
sin = _functional_op(np.sin)
tan = _functional_op(np.tan)

arccos = _functional_op(np.arccos)
arcsin = _functional_op(np.arcsin)
arctan = _functional_op(np.arctan)

log = _functional_op(np.log)
log10 = _functional_op(np.log10)

# For compatibility
pi = np.pi
e = np.e
