"""
Mock/fake objects useful mostly to test base classes
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy

from embfile import EmbFile
from embfile.core import AbstractEmbFileReader, EmbFileReader
from embfile.core._file import DEFAULT_VERBOSE
from embfile.initializers import Initializer
from embfile.types import PairsType, VectorType


class MockEmbFileReader(AbstractEmbFileReader):
    def __init__(self, pairs, out_dtype=numpy.float32):
        super().__init__(out_dtype)
        self._pairs = pairs
        self.close_called = False
        # Note: using two independent indexes ensures skip_vector() is called correctly
        self._iterator = iter(pairs)
        self._current_pair = None

    def _reset(self) -> None:
        self._iterator = iter(self._pairs)
        self._current_pair = None

    def _close(self) -> None:
        self.close_called = True

    def _read_word(self) -> str:
        self._current_pair = next(self._iterator)
        return self._current_pair[0]

    def _read_vector(self) -> VectorType:
        return self._current_pair[1]

    def _skip_vector(self) -> None:
        self._current_pair = None


class MockEmbFile(EmbFile):
    DEFAULT_EXTENSION = '.mock'
    _create_kwargs: SimpleNamespace  # kwargs passed to last call of _create()

    def __init__(self, pairs=[('ciao', numpy.array([1, 2]))],
                 path='path/to/mock_file.mock',
                 out_dtype=None,
                 verbose=DEFAULT_VERBOSE):
        out_dtype = numpy.dtype(out_dtype or pairs[0][1].dtype)
        super().__init__(path, out_dtype, verbose)
        self._pairs = pairs
        self.vocab_size = len(pairs)
        self.vector_size = len(pairs[0][1])
        self.close_called = False

    def _close(self) -> None:
        self.close_called = True

    def _reader(self) -> EmbFileReader:
        return MockEmbFileReader(self._pairs, self.out_dtype)

    @classmethod
    def _create(cls, out_path: Path,
                word_vectors: PairsType, vector_size: int,
                vocab_size: Optional[int], compression: Optional[str] = None,
                verbose: bool = True, **kwargs):
        cls._create_kwargs = SimpleNamespace(
            out_path=out_path,
            create_called=True,
            vocab_size=vocab_size,
            vector_size=vector_size,
            pairs=word_vectors,
            kwargs=kwargs)


class MockInitializer(Initializer):
    FILL_VALUE = 33

    def __init__(self, fill_value=FILL_VALUE):
        self.fit_called = False
        self.fill_value = fill_value

    def fit(self, vectors) -> None:
        self.fit_called = True

    def __call__(self, shape) -> VectorType:
        return numpy.full(shape, fill_value=self.fill_value)
