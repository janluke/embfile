__all__ = ['WordVector']

from typing import NamedTuple

import numpy

from embfile.types import VectorType


class WordVector(NamedTuple):
    """ A (word, vector) NamedTuple """
    word: str
    vector: VectorType

    @staticmethod
    def format_vector(arr):
        """ Used by __repr__ to convert a numpy vector to string.
        Feel free to monkey-patch it. """
        return numpy.array2string(arr, separator=', ', precision=4, threshold=5)

    def __repr__(self):
        vec_str = self.format_vector(self.vector)
        return 'WordVector({!r}, {})'.format(self.word, vec_str)
