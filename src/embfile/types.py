"""
Type aliases used in the library
"""
__all__ = ['VectorType', 'DType', 'PairsType', 'PathType']

from pathlib import Path
from typing import Iterable, Mapping, Tuple, Union

import numpy
from numpy import ndarray

VectorType = ndarray  # TODO: understand why I can't use VectorType in most places (mypy)
DType = Union[str, numpy.dtype]
PairsType = Union[Mapping[str, ndarray], Iterable[Tuple[str, ndarray]]]

PathType = Union[str, Path]
