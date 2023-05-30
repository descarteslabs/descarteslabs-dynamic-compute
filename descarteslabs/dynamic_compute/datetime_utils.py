"""Functionality supporting datetime operations and conversions"""

import datetime
from typing import Union

import dateutil


def normalize_datetime(value: Union[str, datetime.date, datetime.datetime]) -> str:
    """Normalizes an input datetime like object to an isoformatted string

    Parameters
    ----------
    value : Union[str, datetime.date, datetime.datetime]
        The datetime like object to be normalized

    Returns
    -------
    str
        An isoformatted string representation of the input
    """

    if isinstance(value, str):
        value = dateutil.parser.parse(value)

    elif not isinstance(value, (datetime.datetime, datetime.date)):
        err_msg = (
            "Datetimes must be either string, datetime.datetime, "
            f"or datetime.date not '{type(value)}'"
        )
        raise TypeError(err_msg)

    return value.isoformat()


def normalize_datetime_or_none(
    value: Union[str, datetime.date, datetime.datetime, None]
) -> Union[str, None]:
    """Helper function which attempts to normalize the input value as a datetime
    if the value is not None, otherwise retruns the value (aka None)

    Parameters
    ----------
    value : Union[str, datetime.date, datetime.datetime, None]
        The value to normalize

    Returns
    -------
    Union[str, None]
        Returns either the normalized datetime as a string or None
    """

    return normalize_datetime(value) if value is not None else value
