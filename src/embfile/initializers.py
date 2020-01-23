"""
Embedding initializers.
"""
__all__ = ['Initializer', 'NormalInitializer', 'normal']

import abc
from typing import Sequence

import numpy
from overrides import overrides

from embfile.errors import IllegalOperation
from embfile.types import VectorType


class Initializer(abc.ABC):
    """
    A random number generator meant to be used with :meth:`~embfile.build_matrix`.
    It can be fit to a sequence of other vectors in order to compute stats to be used for
    generation. When passed to ``build_matrix``, the initializer is fit to the found vectors.

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def fit(self, vectors: numpy.ndarray):
        """
        *(Abstract method)* Computes stats that will be use for generating new vectors.

        Args:
            vectors:
        """

    @abc.abstractmethod
    def __call__(self, shape) -> VectorType:
        """ *(Abstract method)* Generate an array of shape ``shape`` """


class NormalInitializer(Initializer):
    """
    Generates vectors using a normal distribution with the same mean and standard deviation
    of the set of vectors passed to the fit method. When used with
    :meth:`~embfile.build_matrix`, it initializes out-of-file-vocabulary vectors so
    that they have the same mean and deviation of the vectors found in the file.

    If not fit before to generate vectors, it raises IllegalOperation
    """

    def __init__(self):
        self.loc = None
        self.scale = None

    @overrides
    def fit(self, vectors: numpy.ndarray):
        """ Computes mean and standard deviation of the input vectors """
        if len(vectors) == 0:
            raise ValueError('empty array')
        self.loc = numpy.mean(vectors, axis=0)
        self.scale = numpy.std(vectors, axis=0)

    @overrides
    def __call__(self, shape: Sequence[int]):
        if self.loc is None or self.scale is None:
            raise IllegalOperation('you must fit this initializer before you can use it')
        return numpy.random.normal(loc=self.loc, scale=self.scale, size=shape)


def normal(mean=0.0, deviation=None):
    """
    Returns a normal sampler. If deviation is not given, it is set dynamically to

        1.0 / sqrt(shape[-1])

    where shape[-1] is the vector size.
    """

    def generate(shape):
        scale = 1.0 / shape[-1] if deviation is None else deviation
        return numpy.random.normal(loc=mean, scale=scale, size=shape)

    return generate
