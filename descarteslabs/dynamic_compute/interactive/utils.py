from typing import Any, Tuple


def tuple_del(in_tuple: Tuple, index: int) -> Tuple:
    if index < 0:
        index = len(in_tuple) + index
    return in_tuple[:index] + in_tuple[index + 1 :]


def tuple_insert(in_tuple: Tuple, index: int, value: Any) -> Tuple:
    if index < 0:
        index = len(in_tuple) + index + 1
    return in_tuple[:index] + (value,) + in_tuple[index:]


def tuple_replace(in_tuple: Tuple, index: int, value: Any) -> Tuple:
    if index < 0:
        index = len(in_tuple) + index + 1
    return in_tuple[:index] + (value,) + in_tuple[index + 1 :]


def tuple_move(in_tuple: Tuple, old_index: int, new_index: int) -> Tuple:
    lst = list(in_tuple)
    value = lst.pop(old_index)
    lst.insert(new_index, value)
    return tuple(lst)


def wrap_num(
    x: float, range: Tuple[float] = (-180, 180), include_max: bool = True
) -> float:
    """
    Wrap a number x within a range

    Parameters
    ----------
    x
        float
    range, optional
        tuple of 2 values, representing the min and max range, by default (-180,180)
    include_max, optional
        whether the max should be included, by default True
    include_min, optional
        whether the min should be included, by default True

    Returns
    -------
        x wrapped within the min/max range. Useful for coordinate shifts from out of bounds to in bounds.
    """
    max = range[-1]
    min = range[0]
    d = max - min
    if (x == max and include_max) or (min < x < max):
        return x
    return ((x - min) % d + d) % d + min
