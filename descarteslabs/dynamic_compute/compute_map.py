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

from .blob import GRAFTS_NAMESPACE, create_blob_and_upload_data, load_graft_from_blob
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


def array(*args, **kwargs):
    return None


def groupby(*args, **kwargs):
    return None


class ComputeMap(dict):
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
                    ("filter_scenes", filter_scenes),
                    ("stack_scenes", stack_scenes),
                    ("array", array),
                    ("groupby", groupby),
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
        aoi : descarteslabs.geo.GeoContext
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

    def save_to_catalog_blob(
        self,
        name: str,
        description: Optional[str] = None,
        extra_properties: Optional[Dict[str, Union[str, int, float]]] = None,
        readers: Optional[List[str]] = None,
        writers: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Saves this object to catalog as a Blob

        Parameters
        ----------
        name : str
            The name to give the blob in catalog
        description : Optional[str], optional
            A description of the blob, by default None
        extra_properties : Optional[dict[str, Union[str, int, float]]], optional
            Any extra properties to be stored in the blob, by default None
        readers : Optional[list[str]], optional
            A list of emails, orgs, groups, and users to give read access to the blob, by default None
        writers : Optional[list[str]], optional
            A list of emails, orgs, groups, and users to give write access to the blob, by default None
        tags : Optional[List[str]], optional
            A list of tags to assign to the blob

        Returns
        -------
        str
            The id of the blob created
        """
        extra_properties = extra_properties or {}
        extra_properties["graft_type"] = self.__class__.__name__

        blob = create_blob_and_upload_data(
            json.dumps(dict(self)),
            name,
            namespace=GRAFTS_NAMESPACE,
            description=description,
            extra_properties=extra_properties,
            readers=readers,
            writers=writers,
            tags=tags,
        )

        return blob.id

    @classmethod
    def load_from_catalog_blob(cls, name: str) -> Type[ComputeMap]:
        """Loads an dynamic compute type from catalog

        Parameters
        ----------
        name : str
            The name of the blob in catalog

        Returns
        -------
        Type[ComputeMap]
            The loaded object
        """

        return cls.__SUBCLASSES__[cls.__name__](
            load_graft_from_blob(name, cls.__name__)
        )


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


def index_align_args(f: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    """
    Attempt to make indices compatible for arguments to f. Note that this doesn't ensure success,
    rather it is necessary for certain operations.

    Parameters
    ----------
    f: Callable[[Any, Any], Any]
        Function that takes two arguments and applies a binary operation.

    Returns
    -------
    aligned_f: Callable[[Any, Any], Any]
        Function that takes two arguments, tries to ensure that array indices
        are aligned and applies a binary operation.
    """

    def aligned_f(a, b):
        if issubclass(type(a), Number) or issubclass(type(a), Number):
            # One argument is a number, so the binary operation can
            # be applied in a natural way.
            return f(a, b)

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape == b.shape:
                # The arrays have the same shape, again the binary operation
                # can be applied in a natural way.
                return f(a, b)

            if len(a.shape) == 1 and a.shape[0] == b.shape[0]:
                # a is a vector whose length matches the leading dimension of b.
                # apply f to a after it has been reshaped to be compatible with b.
                new_shape = tuple(
                    [a.shape[0]] + [1 for _ in range(len(b.shape) - len(a.shape))]
                )
                return f(a.reshape(new_shape), b)

            if len(b.shape) == 1 and b.shape[0] == a.shape[0]:
                # Same as the previous case, but a and b are reversed.
                new_shape = tuple(
                    [b.shape[0]] + [1 for _ in range(len(a.shape) - len(b.shape))]
                )
                return f(a, b.reshape(new_shape))

            if len(a.shape) == 3 and len(b.shape) == 4 and a.shape[0] == b.shape[1]:
                # a is a Mosaic and b is an ImageStack, and they have the same number of bands.
                return f(a[None, ...], b)

            if len(a.shape) == 4 and len(b.shape) == 3 and a.shape[0] == b.shape[1]:
                # Same as the previous case, but a and b are reversed.
                return f(a, b[None, ...])

        # The above if-statement handle definite numpy broadcast errors. There are obviously
        # cases that don't fit into one of the above cases. We hand those off to f.
        return f(a, b)

    return aligned_f


#
# Mix-ins for a number of math operations
#


class AddMixin:
    def __add__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a + b), op_name="sum"
        )

    def __radd__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a + b), op_name="sum"
        )


class SubMixin:
    def __sub__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a - b), op_name="sub"
        )

    def __rsub__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            other, self, index_align_args(lambda a, b: b - a), op_name="sub"
        )


class MulMixin:
    def __mul__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a * b), op_name="mul"
        )

    def __rmul__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a * b), op_name="mul"
        )


class TrueDivMixin:
    def __truediv__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a / b), op_name="div"
        )

    def __rtruediv__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: b / a), op_name="div"
        )


class FloorDivMixin:
    def __floordiv__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a // b), op_name="floordiv"
        )

    def __rfloordiv__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: b // a), op_name="floordiv"
        )


class SignedMixin:
    def __abs__(self) -> ComputeMap:
        return type(self)(_apply_unary(self, lambda a: abs(a)))

    def __neg__(self) -> ComputeMap:
        return type(self)(_apply_unary(self, lambda a: -a))


class ExpMixin:
    def __pow__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a**b), op_name="exp"
        )

    def __rpow__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: b**a), op_name="exp"
        )


class CompareMixin:
    def __eq__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a == b), op_name="cmp"
        )

    def __ne__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a != b), op_name="cmp"
        )

    def __gt__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a > b), op_name="cmp"
        )

    def __ge__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a >= b), op_name="cmp"
        )

    def __lt__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a < b), op_name="cmp"
        )

    def __le__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a <= b), op_name="cmp"
        )


class LogicalMixin:
    def __and__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a & b), op_name="and"
        )

    def __rand__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a & b), op_name="and"
        )

    def __or__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a | b), op_name="or"
        )

    def __ror__(self, other: Union[Number, np.ndarray, ComputeMap]) -> ComputeMap:
        return binary_op(
            self, other, index_align_args(lambda a, b: a | b), op_name="or"
        )


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
