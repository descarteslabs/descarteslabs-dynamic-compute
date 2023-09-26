"""
Client to help construct mappings following the graft syntax.

This client is solely concerned with _syntax_, not how to expose
user-facing delayed objects that use grafts. For that, see the
Workflows implementation of Proxytypes, which provides a lazy,
strongly-typed proxy object interface, using graft to represent
the state of its objects.

Warning: the graft client is only guaranteed to be thread-safe for the CPython
interpreter! In other interpreter implementations, where `itertools.count` is
non-atomic, multithreaded use of the graft client could produce invalid grafts.

Subtleties of this Python client
================================

The graft Python client implements a slightly stricter version of the
graft specification, adding two restrictions in places where the formal
syntax is more lax.

1. Keys must be stringified integers, or references to builtins.

   Keys are opaque strings in the graft spec. However, to ensure uniquness,
   keys generated by this Python client are monotonically increasing stringified
   integers. This client assumes additional meaning in these keys in the `function_graft`
   function, where the integer value of keys is used to determine if they came from an
   outer scope or not.

   To follow this pattern, use the `guid` and `current_guid` functions to generate keys,
   rather than coming up with your own system.

2. Grafts with a ``"parameters"`` key are considered _function objects_;
   grafts without a ``"parameters"`` key are considered whatever _value_ they return.

   Every graft is a function, by the spec. However, proxy object libraries
   will probably want to let users work with objects, which feels
   more natural than composing functions.

   Therefore, this client considers a graft with no parameters to represent
   the value it returns. A graft with parameters is considered an actual function.

   In practice, in::

    apply_graft("add", {"x": 1, "returns": "x"}, {"y": 2, "returns": "y"})

   the arguments are considered as the values at their ``"returns"`` keys
   (keys ``x`` and ``y``), rather than "add" being a higher-order function which takes
   two other functions. So the result of this call is something like::

    {"x": 1, "y": 2, "res": ["add", "x", "y"], "returns": "ret"}

   where just the keys ``x`` and ``y`` are given to the ``"add"`` function,
   rather than the two whole grafts.

   Compare this to::

    apply_graft(
        "apply_a_func", {"parameters": ["a"], "returns": "a"}, {"x": 1, "returns": "x"}
    )

   which gives something like::

    {
        "0": {"parameters": ["a"], "returns": "a"},
        "x": 1,
        "1": ["apply_a_func", "0", "x"],
        "returns": "1",
    }

   Notice that the function (``{"parameters": ["a"], "returns": "a"}``)
   is inserted as a subgraft under a new, generated key (``"0"``),
   while the graft of the value (``x``) is copied in directly.

   In general, this graft client handles these concerns correctly for
   you---that's what it's for.
"""

import base64
import contextlib
import copy
import itertools
import json
import pickle
from typing import Dict, Hashable, List

import numpy as np
import six

from .. import syntax

NO_INITIAL = "_no_initial_"
PARAM = "__param__"

GUID_COUNTER = itertools.count()
# use `itertools.count()` as a lock-free threadsafe counter,
# which only works due to a CPython implementation detail
# (because `count` is implemented in C, holding the GIL for the entirety of its execution)
# https://mail.python.org/pipermail//python-ideas/2016-August/041871.html
# https://stackoverflow.com/a/27062830/10519953


def guid():
    return str(next(GUID_COUNTER))


def is_delayed(x):
    "Whether x is a delayed-like: ``x.graft`` is a graft-like mapping"
    try:
        return syntax.is_graft(x.graft)
    except AttributeError:
        return False


def value_graft(value, key=None):
    """
    The graft, as a dict, for a value.

    Parameters
    ----------
    value: delayed-like object or JSON-serializable value or numpy array.
        If a JSON-serializable value, returns the graft representing that value
        (a function with no parameters that returns the value). If a numpy array,
        returns a graft containing an "array" primitive and the serialized array.

        If a delayed-like object, returns ``value.graft``.

    Returns
    -------
    graft: dict
    """
    if is_delayed(value):
        return value.graft
    if key is None:
        key = guid()
    elif not syntax.is_key(key):
        raise TypeError(
            "Key must be a string, and not one of {}".format(syntax.RESERVED_WORDS)
        )
    if isinstance(value, syntax.PRIMITIVE_TYPES):
        return {key: value, "returns": key}
    elif isinstance(value, (tuple, list, dict)):
        # Quoted JSON
        return {key: [value], "returns": key}
    elif isinstance(value, np.ndarray):
        key1 = guid()
        return {
            "returns": key,
            key: ["array", key1],
            key1: base64.b64encode(pickle.dumps(value)).decode("utf-8"),
        }
    else:
        raise TypeError(
            "Value must be a delayed-like object, primitve (one of {}), or JSON-serializable "
            "sequence or mapping, not {}: {}".format(
                syntax.PRIMITIVE_TYPES + (np.ndarray,), type(value), value
            )
        )


def keyref_graft(key):
    """
    Graft for referring to an arbitrary key.

    Useful for referring to parameters of functions, or builtins.

    Parameters
    ----------
    key: str
    """
    if not syntax.is_key(key):
        raise ValueError(
            "Key must be a string, and not one of {}".format(syntax.RESERVED_WORDS)
        )

    return {"returns": key}


def is_keyref_graft(value):
    return syntax.is_graft(value) and len(value) == 1 and next(iter(value)) == "returns"


def compress_graft(graft: Dict) -> Dict:
    """
    Given a graft return a new graft that removes redundant entries.

    Parameters
    ----------
    graft: Dict
        Graft to be compressed. Note that this graft is not changed, rather a new
        graft is returned if the input graft can be compressed.

    Returns
    -------
    new_graft: Dict
        Either the original graft instance, or a new compressed graft
    """

    # Create a map from the grafts' string values (other than "return") back to the key(s)
    # where the strings occur. This is an reverse (multi) map of the graft.
    rev_graft = {}
    for key in graft:
        if key == "returns":
            continue

        # graft[k] may not be hashable, but it's serializable.
        rev_graft.setdefault(json.dumps(graft[key]), []).append(key)

    # Create a map called replacements from a key to be replaced to its replacement
    replacements = {}
    for keys in rev_graft.values():
        # keys is a list of keys that all point to the same value.

        if len(keys) < 2:
            # There is only one key pointing to this value.
            continue

        # Record that we can replace all successive keys with the first
        # key in the list
        for key in keys[1:]:
            replacements[key] = keys[0]

    # If there is nothing to replace, return the graft as it is.
    if not replacements:
        return graft

    # Create a new copy of the graft
    new_graft = copy.deepcopy(graft)

    # Walk the graft and replace places where we *use* the key to be replaced
    for key in new_graft:

        if isinstance(new_graft[key], list):

            # Here op_list is something like, ['code', '73', '72', {'cache_id': '75'}]
            op_list = new_graft[key]

            for i, item in enumerate(op_list):
                if i == 0:
                    # This is the primitive type, e.g. 'code'
                    continue

                if isinstance(item, str):
                    # The following will update op_list[i] with its replacement if
                    # there is one, otherwise it will leave op_list[i] as it is
                    op_list[i] = replacements.get(item, item)
                elif isinstance(item, dict):
                    # Here item is something like {'cache_id': '75'}
                    for key1 in item:
                        item[key1] = replacements.get(item[key1], item[key1])

    # Remove the keys are no longer referenced
    for key in replacements:
        _ = new_graft.pop(key)

    # We've modified the graft, perform another pass to see if more compression is possible.
    return compress_graft(new_graft)


def normalize_graft(graft: Dict) -> Dict:
    """
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

    sorted_non_return_keys = sorted(filter(lambda key: key != "returns", graft.keys()))

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


def splice(graft1: dict, splice_value: Hashable, graft2: dict) -> dict:
    """
    Splice two grafts together by replacing a particular value in graft1 with
    the returned content of graft2.

    For example
    >>> splice(
    >>>     {
    >>>       "returns": "0",
    >>>       "0": ["1", "2"],
    >>>       "1": "splice value",
    >>>       "2": "other value"
    >>>     },
    >>>     "splice value",
    >>>     {
    >>>       "returns": "3",
    >>>       "3": "new value"
    >>>     }
    >>> )
    returns
    >>>     {
    >>>       "returns": "0",
    >>>       "0": ["3", "2"],
    >>>       "3": "new value",
    >>>       "2": "other value"
    >>>     }

    Parameters
    ----------
    graft1: dict
        Graft into which graft2 will be spliced. Note that if splice_value is not
        a value in graft1, graft1 will be returned.
    splice_value: Hashable
        Value in graft1 to replace with the "return" content from graft2
    graft2: dict
        Graft to be spliced into graft1, must containt a "return" key.

    Returns
    -------
    """

    assert syntax.is_graft(graft1)
    assert syntax.is_graft(graft2)
    keyset1 = set(graft1.keys())
    keyset2 = set(graft2.keys())
    assert "returns" in keyset2
    assert (keyset1 - {"returns"}).isdisjoint(keyset2)

    keys_pointing_to_splice_value = [
        key for key in graft1 if graft1[key] == splice_value
    ]

    spliced_graft = copy.deepcopy(graft1)
    for key in keys_pointing_to_splice_value:
        spliced_graft.pop(key)

    graft2 = copy.deepcopy(graft2)
    splice_key = graft2.pop("returns")

    key_remap = {key: splice_key for key in keys_pointing_to_splice_value}

    for key in spliced_graft:
        value = spliced_graft[key]

        if isinstance(value, list):
            value_as_list = value
            for i, object in enumerate(value_as_list):

                if isinstance(object, str):
                    value_as_list[i] = key_remap.get(object, object)

                if isinstance(object, dict):
                    object_as_dict = object
                    for subkey in object_as_dict:
                        object_as_dict[subkey] = key_remap.get(
                            object_as_dict[subkey], object_as_dict[subkey]
                        )

    spliced_graft.update(graft2)

    return spliced_graft


def op_args(op_list: List) -> List:
    """
    Given a list that encodes an operation and its arguments return the arguments

    Parameters
    ----------
    op_list: List
        List that encodes an operation and its arguments

    Returns
    -------
        Arguments
    """
    return op_list[1:]


def find_unreferenced(graft: dict) -> List[str]:
    """
    Given a graft, return a list of unreferenced keys.

    Parameters
    ----------
    graft: dict
        Graft in which we should count references.

    Returns
    -------
    unreferences: List[str]
        List of keys in the graft that are not referenced.
    """

    keys = graft.keys()
    count = {key: 0 for key in keys}
    for graft_value in graft.values():

        if not isinstance(graft_value, list):
            continue

        for obj in op_args(graft_value):
            if isinstance(obj, str) and obj in count:
                count[obj] += 1

            elif isinstance(obj, dict):
                for obj_value in obj.values():
                    if obj_value in count:
                        count[obj_value] += 1

    return [key for key in count if count[key] == 0]


def unset_all_cache_ids(graft: dict) -> dict:
    """
    Given a graft, return a new graft that has no cache IDs set.

    Parameters
    ----------
    graft: dict
        Graft possibly containing graft IDs

    Returns:
    new_graft: dict
        Copy of input graft without graft IDs.
    """

    new_graft = copy.deepcopy(graft)

    cache_ids = []
    for graft_value in new_graft.values():

        if not isinstance(graft_value, list):
            continue

        objs_to_remove = []
        for obj in op_args(graft_value):

            if not isinstance(obj, dict):
                continue

            for obj_key, obj_value in obj.items():
                if obj_key == "cache_id":
                    cache_ids.append(obj_value)
                    break

            obj.pop("cache_id", None)
            if len(obj) == 0:
                objs_to_remove.append(obj)

        for obj_to_remove in objs_to_remove:
            graft_value.remove(obj_to_remove)

    unreferenced = find_unreferenced(new_graft)
    for cache_id in cache_ids:
        if cache_id in unreferenced:
            new_graft.pop(cache_id, None)

    return new_graft


def apply_graft(function, *args, **kwargs):
    """
    The graft for calling a function with the given positional and keyword arguments.

    Arguments can be given as Python values, in which case `value_graft`
    will be called on them first, or as delayed-like objects or graft-like mappings.

    Parameters
    ----------
    function: str, graft-like mapping, or delayed-like object
        The function to apply
    **args: delayed-like object, graft-like mapping, or JSON-serializable value
        Positional arguments to apply function to
    **kwargs: delayed-like object, graft-like mapping, or JSON-serializable value
        Named arguments to apply function to

    Returns
    -------
    result_graft: dict
        Graft representing ``function`` applied to ``args`` and ``kwargs``
    """
    pos_args_grafts = [
        arg if syntax.is_graft(arg) else value_graft(arg) for arg in args
    ]
    named_arg_grafts = {
        name: (arg if syntax.is_graft(arg) else value_graft(arg))
        for name, arg in six.iteritems(kwargs)
    }

    if is_delayed(function):
        function = function.graft

    result_graft = {}
    function_key = None
    if isinstance(function, str):
        function_key = function
    elif syntax.is_graft(function):
        if "parameters" in function:
            # function considered an actual function object, insert it as a subgraft
            param_names = function.get("parameters", [])
            syntax.check_args(len(args), six.viewkeys(kwargs), param_names)

            function_key = guid()
            result_graft[function_key] = function
        else:
            # function considered the value it returns; inline its graft.
            # this is the case with higher-order functions,
            # where `function` is an apply expression that returns another function.
            # we don't check args because that would require interpreting the graft.
            result_graft.update(function)
            function_key = function["returns"]
    else:
        raise TypeError(
            "Expected a graft dict, a delayed-like object, or a string as the function; "
            "got {}".format(function)
        )

    positional_args = []
    named_args = {}
    for name, arg_graft in itertools.chain(
        zip(itertools.repeat(None), pos_args_grafts), six.iteritems(named_arg_grafts)
    ):
        if graft_is_function_graft(arg_graft):
            # argument considered an actual function object, insert it as a subgraft
            arg_key = guid()
            result_graft[arg_key] = arg_graft
        else:
            # argument considered the value it returns; inline its graft
            result_graft.update(arg_graft)
            arg_key = arg_graft["returns"]

        if name is None:
            positional_args.append(arg_key)
        else:
            named_args[name] = arg_key

    expr = [function_key] + positional_args
    if len(named_args) > 0:
        expr.append(named_args)

    key = guid()
    result_graft[key] = expr
    result_graft["returns"] = key
    return compress_graft(result_graft)


def is_function_graft(graft):
    return syntax.is_graft(graft) and graft_is_function_graft(graft)


def graft_is_function_graft(graft):
    return "parameters" in graft


def function_graft(result, *parameters, **kwargs):
    """
    Graft for a function that returns ``result``.

    Parameters
    ----------
    result: graft-like mapping or delayed-like object
        The value to be returned by the function.

        If a value graft (no ``"parameters"`` key), produces a function
        that returns that value.

        If a function graft (has ``"parameters"``), produces a higher-order
        function (function that itself returns a function).
    *parameters: str or keyref graft
        Names of the parameters to the function, or keyref grafts representing them.
        The graft of ``result`` should include dependencies to these names
        (using `keyref_graft`), but this is not validated. Forgetting to include
        a parameter name required somewhere within ``result`` could result
        in unexpected runtime behavior.
    first_guid: optional, str or None, default None
        The value of `guid` when the logical scope of this function begins.
        If given, any keys that show up in ``result`` that strictly precede ``first_guid``
        (when compared as ints, not lexicographically) are assumed to be references
        to an outer scope level, and are moved outside the body of the function.
        This is important to use when delaying Python functions by passing dummy
        delayed-like objects through them, since otherwise references to objects in
        outer scopes in Python will get "inlined" into the functions graft, which could
        have performance implications if the function is called repeatedly.

        Note that this only searches the keys of ``result``, and does not traverse
        into any sub-grafts it might contain, so this doesn't work on higher-order
        functions. If ``result`` itself returns a sub-function, and that sub-function
        references a key preceding ``first_guid``, it won't be removed from the sub-function's
        graft.

    Returns
    -------
    function_graft: dict
        Graft representing a function that returns ``result`` and takes ``parameters``.
    """
    first_guid = kwargs.pop("first_guid", None)
    if len(kwargs) > 0:
        raise TypeError(
            "Unexpected keyword arguments {}".format(", ".join(map(repr, kwargs)))
        )
    if first_guid is not None:
        first_guid = int(first_guid)

    parameters = [
        param["returns"] if is_keyref_graft(param) else param for param in parameters
    ]
    if not syntax.is_params(parameters):
        raise ValueError(
            "Invalid parameters for a graft (must be a sequence of strings, "
            "none of which are in {}): {}".format(syntax.RESERVED_WORDS, parameters)
        )
    result_graft = result if syntax.is_graft(result) else value_graft(result)

    if first_guid is not None:
        orig_result_graft = result_graft
        result_graft = orig_result_graft.copy()
        containing_scope = {
            k: result_graft.pop(k)
            for k in orig_result_graft
            if _is_outer_scope(k, first_guid)
        }
    else:
        containing_scope = {}

    if graft_is_function_graft(result_graft):
        # Graft that returns a function object; i.e. has a "parameters" key
        key = guid()
        containing_scope.update(
            {"parameters": parameters, key: result_graft, "returns": key}
        )
        return containing_scope
    else:
        # Graft that returns the value referred to by result
        func_graft = dict(result_graft, parameters=parameters)
        if len(containing_scope) == 0:
            return func_graft
        else:
            key = guid()
            containing_scope[key] = func_graft
            containing_scope["returns"] = key
            return containing_scope


def _is_outer_scope(key, first_guid):
    try:
        return int(key) < first_guid
    except ValueError:
        return False


def merge_value_grafts(**grafts):
    """
    Merge zero-argument grafts into one, with return values available under new names.

    Lets you take multiple grafts that return values (such as ``{'x': 1, 'returns': 'x'}``),
    and construct a graft in which those _returned_ values are available under the names
    specified as keyword arguments---as _values_, not as callables.

    Parameters
    ----------
    **grafts: delayed-like object, graft-like mapping, or JSON-serializable value
        Grafts that take no arguments: delayed-like objects with no dependencies on parameters,
        JSON-serializable values, or grafts without parameters.
        The value _returned_ by each graft will be available as the name given
        by its keyword argument.
        Except for JSON-serializable values, each graft will be kept as a sub-graft within its own scope,
        so overlapping keys between the grafts will not collide.
        Caution: this function accepts both grafts and JSON values, so be careful
        that you do not pass in a JSON value that looks like a graft, since it will not get quoted.
    """
    merged = {}
    for name, value in six.iteritems(grafts):
        if name in syntax.RESERVED_WORDS:
            raise ValueError(
                "Cannot use reserved name {!r} as a key in a graft".format(name)
            )
        if isinstance(value, syntax.PRIMITIVE_TYPES):
            # fastpath for simple case
            merged[name] = value
        else:
            subgraft = value if syntax.is_graft(value) else value_graft(value)
            parameters = subgraft.get("parameters", ())

            if len(parameters) > 0:
                raise ValueError(
                    "Value graft for {}: expected a graft that takes no parameters, "
                    "but this one takes {}".format(name, parameters)
                )
            try:
                returned = subgraft[subgraft["returns"]]
            except KeyError as e:
                raise KeyError(
                    "In subgraft {!r}: returned key {} is undefined".format(name, e)
                )
            if syntax.is_literal(returned) or syntax.is_quoted_json(returned):
                merged[name] = returned
            else:
                # insert actual subgraft under a different name
                subkey = "_{}".format(name)
                merged[subkey] = subgraft
                # actual name is the invocation of that subgraft, with no arguments
                merged[name] = [subkey, {}]
    return merged


def isolate_keys(graft, wrap_function=False):
    """
    Isolate a value graft to its own subscope, to prevent key collisions.

    If ``graft`` already uses valid GUID keys, this ensures that subsequent `apply_graft`
    operations on ``graft`` won't collide with its existing keys.
    Essentially, "key-namespace isolation".

    Usually a value graft, i.e. a graft with no ``"parameters"`` key
    that refers to the original value of ``graft``. See the ``wrap_function``
    argument for details.

    Parameters
    ----------
    graft: graft-like mapping or delayed-like object
        The graft or delayed object to scope-isolate.
    wrap_function: bool, optional, default False
        If a function graft is given (contains a ``"parameters"`` key),
        whether to wrap it in an outer graft.

        Usually, this is not necessary, since `apply_graft` would never
        add additional keys to a function graft, making collisions with new
        keys impossible. In some cases though, it may be preferable to always
        get a value graft back, regardless of whether ``graft`` was a function
        graft or a value graft.

        If False (default), function grafts are returned unmodified.

        If True and ``graft`` is a function graft, a value graft is returned
        that refers to that function.

    Returns
    -------
    isolated_graft: dict
        Value graft representing the same value as ``graft``,
        but isolated to a subscope. Or, if ``graft`` was a function graft,
        and ``wrap_function`` is False, it's returned unmodified.
    """
    graft = graft if syntax.is_graft(graft) else value_graft(graft)

    if graft_is_function_graft(graft):
        if not wrap_function:
            return graft
        else:
            subgraft_key = guid()
            return {subgraft_key: graft, "returns": subgraft_key}
    else:
        subgraft_key = guid()
        result_key = guid()
        return {subgraft_key: graft, result_key: [subgraft_key], "returns": result_key}


def parametrize(graft, **params):
    """
    Isolate ``graft`` to its own subscope, in which ``params`` are defined.

    Parameters
    ----------
    graft: graft-like mapping or delayed-like object
        The graft or delayed object to scope-isolate and parametrize
    **params: delayed-like object, graft-like mapping, or JSON-serializable value
        Grafts that take no arguments: delayed-like objects with no dependencies on parameters,
        JSON-serializable values, or grafts without parameters.

        The value _returned_ by each graft will be bound to the name given
        by its keyword argument within the scope in which ``graft`` executes.
        These names will _not_ be defined within the top-level scope of the returned graft,
        unless none of the names are valid GUIDs, in which case they are put in
        the outer scope since is no collison risk.

        Except for JSON-serializable values, each parameter will be kept as a sub-graft within its own scope,
        so overlapping keys between the parameters will not collide.
        Caution: this function accepts both grafts and JSON values, so be careful
        that you do not pass in a JSON value that looks like a graft, since it will not get quoted.

    Returns
    -------
    parametrized_graft: dict
        Value graft also representing ``graft``,
        but with ``graft`` isolated to a subscope in which ``params`` are defined.
    """
    for param in params:
        if param in syntax.RESERVED_WORDS:
            raise ValueError(
                "Cannot use reserved name {!r} as a key in a graft".format(param)
            )

    subgraft = isolate_keys(graft, wrap_function=True)
    subgraft.update(merge_value_grafts(**params))

    if any(syntax.is_guid_key(param) for param in params):
        return isolate_keys(subgraft)
    else:
        return subgraft


@contextlib.contextmanager
def consistent_guid(start=0):
    """
    Context manager or decorator to temporarily reset the GUID, for use in testing to ensure consistent grafts

    Not thread-safe.
    """
    global GUID_COUNTER
    original_counter = GUID_COUNTER
    if getattr(consistent_guid, "_in_use", False):
        raise RuntimeError(
            "consistent_guid is already in use and cannot be used reentrantly"
        )
    try:
        consistent_guid._in_use = True
        GUID_COUNTER = itertools.count(start)
        yield
    finally:
        GUID_COUNTER = original_counter
        consistent_guid._in_use = False
