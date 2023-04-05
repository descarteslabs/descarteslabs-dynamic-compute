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
