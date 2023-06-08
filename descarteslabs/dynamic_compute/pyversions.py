"""Functionality for dealing with python versions"""

from __future__ import annotations

import sys
from typing import Optional, Union


def _dot_joins(*args):
    return ".".join([str(val) for val in args])


def _specifier_or_zero(values: list[int], idx: int) -> int:
    try:
        return values[idx]
    except IndexError:
        return 0


class PythonVersion:
    def __init__(
        self,
        major: int,
        minor: Optional[int] = None,
        micro: Optional[int] = None,
    ):
        """
        Representation of a python version

        Parameters
        ----------
        major : int
            The major python version
        minor : Optional[int]
            The minor python version, defaults to 0
        micro : Optional[int]
            The micro python version, defaults to 0
        """

        self.major = major
        self.minor = minor or 0
        self.micro = micro or 0

    def __eq__(self, other: Union[str, PythonVersion]):
        if isinstance(other, str):
            other = PythonVersion.from_string(other)

        return (
            self.major == other.major
            and self.minor == other.minor
            and self.micro == other.micro
        )

    @classmethod
    def from_sys(cls) -> PythonVersion:
        """Constructor from system information"""

        return cls(
            major=sys.version_info.major,
            minor=sys.version_info.minor,
            micro=sys.version_info.micro,
        )

    @classmethod
    def from_string(cls, version_string: str) -> PythonVersion:
        """Constructor from a string

        Parameters
        ----------
        version_string : str
            The version string. This must include at least a major version

        Returns
        -------
        PythonVersion
            The parsed PythonVersion object
        """
        vals = [int(val) for val in version_string.split(".")]

        if len(vals) > 3:
            err_msg = "String must contain only major, minor, and micro versions"
            raise ValueError(err_msg)

        return cls(
            major=vals[0],
            minor=_specifier_or_zero(vals, 1),
            micro=_specifier_or_zero(vals, 2),
        )

    @property
    def major_minor(self) -> str:
        """major.minor string representation of this version"""

        return _dot_joins(self.major, self.minor)

    @property
    def major_minor_micro(self) -> str:
        """major.minor.micro string representation of this version"""

        return _dot_joins(self.major_minor, self.micro)

    def compatible_with(self, version: Union[str, PythonVersion]) -> bool:
        """Checks that this version is compatible with the provided version

        Parameters
        ----------
        version : Union[str, PythonVersion]
            Either a string or PythonVersion

        Returns
        -------
        bool
            True if versions are compatible, False otherwise
        """

        if isinstance(version, str):
            version = PythonVersion.from_string(version)

        return self.major == version.major and self.minor == version.minor
