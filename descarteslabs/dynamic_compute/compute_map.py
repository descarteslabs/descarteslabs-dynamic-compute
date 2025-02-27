# This can be remoted with Python 3.11
# https://peps.python.org/pep-0673/
from __future__ import annotations

import json
import sys
from abc import ABC, abstractclassmethod, abstractmethod
from copy import copy, deepcopy
from io import StringIO
from numbers import Number
from typing import Dict, List, Type, Union

import descarteslabs as dl
import numpy as np

from .graft.client import client as graft_client
from .graft.interpreter.interpreter import interpret
from .graft.syntax import syntax as graft_syntax
from .operations import (
    _func_op,
    _math_op,
    _resolution_graft_x,
    _resolution_graft_y,
    compute_aoi,
)


class DotDict(dict):
    """
    dot-notation access to top-level dictionary attributes

    From https://stackoverflow.com/a/23689767
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


type_map = {"int": int}


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


def stack_scenes(*args, **kwargs):
    return None


def array(*args, **kwargs):
    return None


def groupby(*args, **kwargs):
    return None


def filter_data(*args, **kwargs):
    return None


def groupby_data(*args, **kwargs):
    return None


def math_op(*args, **kwargs):
    return None


def reduction_op(*args, **kwargs):
    return None


def clip_data(*args, **kwargs):
    return None


def fill_mask(*args, **kwargs):
    return None


def graft_resolution_x(*args, **kwargs):
    return None


def graft_resolution_y(*args, **kwargs):
    return None


def band_op(*args, **kwargs):
    return None


def index(*args, **kwargs):
    return None


def length(*args, **kwargs):
    return None


def mask(*args, **kwargs):
    return None


def functional(*args, **kwargs):
    return None


def dot(*args, **kwargs):
    return None


class ComputeMap(dict, ABC):
    """
    A wrapper class to support operations on grafts. Proxy objects should all be
    descended from ComputeMap

    It is natural to apply a binary operation to operands of different types.
    For example, we may want to add a constant to a subclass of `ComputeMap`.
    In this case the intent is to add that constant to every value in the array
    to which the subclass of ComputeMap evaluates. The result of this addition
    operation should be the subclass of `ComputeMap`.

    It is more complicated when the operands are both subclasses of `ComputeMap`,
    but neither is a subclass of the other. e.g. `Mosaic` and `ImageStack`. In this
    case, we apply the operation to the `Mosaic` instance and each image in the
    `ImageStack` instance. The return value is of type `ImageStack`.

    This logic cannot be inferred by type _relationships_. `ImageStack` and `Mosaic`
    are both subclasses of `ComputeMap` and neither is descended from the other.
    Code would require explict reference to types, which if present here would lead
    to a circular dependency.

    In response, `ComputeMap` and its subclasses provide a `_RETURN_PRECEDENCE` class
    attribute that allows one to determine which operand type should take precedence
    for the return type of a binary operation. In particular it supports code like

    ```
    if t1._RETURN_PRECEDENCE > t2._RETURN_PRECEDENCE:
        return t1

    return t2
    ```

    Thereby allowing operations on types without explicit reference to the types.
    The logic assumes that there is an ordering for subclasses of `ComputeMap`.
    """

    _RETURN_PRECEDENCE = 0
    __SUBCLASSES__: Dict[str, Type[ComputeMap]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ComputeMap.__SUBCLASSES__[cls.__name__] = cls

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
                    ("stack_scenes", stack_scenes),
                    ("array", array),
                    ("groupby", groupby),
                    ("filter_data", filter_data),
                    ("groupby_data", groupby_data),
                    ("math", math_op),
                    ("reduction", reduction_op),
                    ("clip", clip_data),
                    ("resolution_y", graft_resolution_y),
                    ("resolution_x", graft_resolution_x),
                    ("band_op", band_op),
                    ("index", index),
                    ("length", length),
                    ("mask", mask),
                    ("functional", functional),
                    ("dot", dot),
                    ("filled", fill_mask),
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
        self.init_args = {}

    def __getattr__(self, attr):
        # Provide a way to evaluate to *just* the raster data or *just* the properties.
        # This is in support of a Workflows like interface.
        if attr not in ["properties", "ndarray"]:
            try:
                return super().__getattr__(self, attr)
            except:  # noqa E722
                raise AttributeError(f"{attr} is not supported by dynamic-compute")

        new_compute_map = copy(self)
        new_compute_map.return_val = attr
        return new_compute_map

    def compute(
        self, aoi: dl.geo.AOI, **kwargs
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
        aoi : descarteslabs.geo.GeoContext
            GeoContext for which to compute evaluate this ComputeMap

        Returns
        -------
        results : Union[Array, List, Dict, DotDict]
            Evaluation of self for this AOI. The return will be either
            an array, properties as a list, properties as dict, or both
            as a DotDict
        """

        value, properties = compute_aoi(self, aoi, **kwargs)

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

    @abstractmethod
    def serialize(self):
        """Abstract method for serializing this object's state"""

        err_msg = "Abstract method, must be implemented by subclasses"
        raise NotImplementedError(err_msg)

    @abstractclassmethod
    def deserialize(cls):
        """Abstract method for deserializing state into an instance of this object"""

        err_msg = "Abstract method, must be implemented by subclasses"
        raise NotImplementedError(err_msg)

    def _extra_init_args(self):
        """
        An instance of a ComputeMap is initialized from a dict. Instances
        of subclasses of ComputeMap sometimes require additional arguments to
        initialize.

        Returns
        -------
        extra_args: dict
            A copy of any extra arguments that should be passed to the constructor to recreate "self"
        """
        return deepcopy(self.init_args)


def as_compute_map(a: Union[Number, Dict, ComputeMap, np.ndarray, List]) -> ComputeMap:
    """
    Return the input as a ComputeMap or raise an exception if this isn't possible.

    Parameters
    ----------
    a : Union[ComputeMap, Number, Dict, numpy.ndarray, List]
        Value to be represented as a ComputeMap

    Return
    ------
    cma : ComputeMap
        Input a as a ComputeMap instance.
    """

    if isinstance(a, Number) or isinstance(a, np.ndarray) or isinstance(a, list):
        if isinstance(a, list):
            a = np.array(a)
        return ComputeMap(graft_client.value_graft(a))

    if not isinstance(a, ComputeMap):
        # `a` is a Dict, but not a ComputeMap. If `a` it isn't a graft,
        # the ComputeMap constructor will throw the necessary exception.
        return ComputeMap(a)

    return a


def resolution_x():
    """
    Generate ComputeMap representation of resolution_x.

    Returns
    -------
    Resolution: ComputeMap
        A dynamic-compute object representation of resolution east-west
    """
    return as_compute_map(_resolution_graft_x())


def resolution_y():
    """
    Generate ComputeMap representation of resolution_y.

    Returns
    -------
    Resolution: ComputeMap
        A dynamic-compute object representation of resolution n-s.
    """
    return as_compute_map(_resolution_graft_y())


def type_max(
    t1: Union[np.ndarray, ComputeMap], t2: Union[np.ndarray, ComputeMap]
) -> Union[np.ndarray, ComputeMap]:
    """
    Return the more general of two types. If either t1 or t2 is a number or array
    return the other. If neither is a number or array and one is descended from
    the other return the ancestor type. If neither is a number or array and
    neither is descended from the other use the _RETURN_PRECEDENCE attribute.

    This is used in a default implementation for computing the return types of binary
    operations. This may be insufficient for certain future classes, but can be
    overridden for classes in which this is the case.

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

    if t1._RETURN_PRECEDENCE > t2._RETURN_PRECEDENCE:
        return t1

    return t2


#
# Mix-ins for a number of math operations
#


def _extra_init_args(cm):
    # Get extra arguments used with the compute map constructor, if available
    if issubclass(type(cm), ComputeMap):
        return cm._extra_init_args()
    return {}


class AddMixin:
    def __add__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "add", as_compute_map(other)))

    def __radd__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "radd", as_compute_map(other)))


class SubMixin:
    def __sub__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "sub", other))

    def __rsub__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "rsub", as_compute_map(other)))


class MulMixin:
    def __mul__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "mul", as_compute_map(other)))

    def __rmul__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "rmul", as_compute_map(other)))


class TrueDivMixin:
    def __truediv__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "truediv", as_compute_map(other)))

    def __rtruediv__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "rtruediv", as_compute_map(other)))


class FloorDivMixin:
    def __floordiv__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "floordiv", as_compute_map(other)))

    def __rfloordiv__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "rfloordiv", as_compute_map(other)))


class SignedMixin:
    def __abs__(self) -> ComputeMap:

        # extra_init_args = _extra_init_args(self)
        return_type = type(self)
        return return_type(_math_op(self, "_abs"))

    def __neg__(self) -> ComputeMap:

        # extra_init_args = _extra_init_args(self)
        return_type = type(self)
        return return_type(_math_op(self, "neg"))


class ExpMixin:
    def __pow__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "_pow", as_compute_map(other)))

    def __rpow__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "rpow", as_compute_map(other)))


class CompareMixin:
    def __eq__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "eq", as_compute_map(other)))

    def __ne__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "ne", as_compute_map(other)))

    def __gt__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "gt", as_compute_map(other)))

    def __ge__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "ge", as_compute_map(other)))

    def __lt__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "lt", as_compute_map(other)))

    def __le__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "le", as_compute_map(other)))


class LogicalMixin:
    def __and__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "_and", as_compute_map(other)))

    def __rand__(
        self, other: Union[Number, List, np.ndarray, ComputeMap]
    ) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "rand", as_compute_map(other)))

    def __or__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "_or", as_compute_map(other)))

    def __ror__(self, other: Union[Number, List, np.ndarray, ComputeMap]) -> ComputeMap:

        return_type = type_max(type(self), type(other))
        return return_type(_math_op(self, "ror", as_compute_map(other)))

    def __invert__(self) -> ComputeMap:

        return_type = type(self)

        return return_type(_math_op(self, "invert"))


class NumpyReductionMixin:
    def max(self, axis):
        """
         Apply np.ma.max to the ComputeMap

        Args:
            axis (str): Axis over which to call the reducer. If called on a Mosaic, must be in ["bands"].

        Returns:
            ComputeMap
        """
        return self.reduce("max", axis)

    def mean(self, axis):
        """
         Apply np.ma.mean to the ComputeMap

        Args:
            axis (str): Axis over which to call the reducer. If called on a Mosaic, must be in ["bands"].

        Returns:
            ComputeMap
        """
        return self.reduce("mean", axis)

    def median(self, axis):
        """
         Apply np.ma.median to the ComputeMap

        Args:
            axis (str): Axis over which to call the reducer. If called on a Mosaic, must be in ["bands"].

        Returns:
            ComputeMap
        """
        return self.reduce("median", axis)

    def min(self, axis):
        """
         Apply np.ma.min to the ComputeMap

        Args:
            axis (str): Axis over which to call the reducer. If called on a Mosaic, must be in ["bands"].

        Returns:
            ComputeMap
        """
        return self.reduce("min", axis)

    def sum(self, axis):
        """
         Apply np.ma.sum to the ComputeMap

        Args:
            axis (str): Axis over which to call the reducer. If called on a Mosaic, must be in ["bands"].

        Returns:
            ComputeMap
        """
        return self.reduce("sum", axis)

    def std(self, axis):
        """
         Apply np.ma.std to the ComputeMap

        Args:
            axis (str): Axis over which to call the reducer. If called on a Mosaic, must be in ["bands"].

        Returns:
            ComputeMap
        """
        return self.reduce("std", axis)

    def argmax(self, axis):
        """
            Apply np.ma.argmax to the ComputeMap

        Args:
            axis: (str)
                Axis over which to call the reducer. If called on a Mosaic, must be in ["bands"].
                If called on an ImageStack, must be in ["bands", "images"]

        Returns:
            Mosaic
        """
        if axis not in ["bands", "images"]:
            raise NotImplementedError(f"argmax reduction over {axis} not implemented")
        return self.reduce("argmax", axis)

    def argmin(self, axis):
        """
            Apply np.ma.argmin to the ComputeMap

        Args:
            axis: (str)
                Axis over which to call the reducer. If called on a Mosaic, must be in ["bands"].
                If called on an ImageStack, must be in ["bands", "images"]

        Returns:
            Mosaic
        """
        if axis not in ["bands", "images"]:
            raise NotImplementedError(f"argmax reduction over {axis} not implemented")
        return self.reduce("argmin", axis)


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

        return type(arg)(_func_op(arg, f))

    return op


#
# functional operations
#
def arctan2(x, y):
    return _math_op(x, "arctan2", y)


sqrt = _functional_op("sqrt")

cos = _functional_op("cos")
sin = _functional_op("sin")
tan = _functional_op("tan")

arccos = _functional_op("arccos")
arcsin = _functional_op("arcsin")
arctan = _functional_op("arctan")

log = _functional_op("log")
log10 = _functional_op("log10")

# For compatibility
pi = np.pi
e = np.e
