import abc

import numpy
from overrides import overrides

from embfile.errors import IllegalOperation
from embfile.types import DType, VectorType


class EmbFileReader(abc.ABC):
    """
    *(Abstract class)* Iterator that yields a word at each step and read the corresponding vector
    only if the lazy property ``current_vector`` is accessed.

    **Iteration model.** The iteration model is not the most obvious: each iteration step doesn't
    return a word vector pair. Instead, for performance reasons, at each step a reader returns
    the next word. To read the vector for the current word, you must access the (lazy) property
    :meth:`current_vector`::

        with emb_file.reader() as reader:
            for word in reader:
                if word in my_vocab:
                    word2vec[word] = reader.current_vector

    When you access :meth:`~embfile.core.EmbFileCursor.current_vector` for the first time,
    the vector data is read/parsed and a vector is created; the vector remains
    accessible until a new word is read.

    **Creation.** Creating a reader usually implies the creation of a file object. That's why
    ``EmbFileReader`` implements the ``ContextManager`` interface so that you can use it inside
    a ``with`` clause. Nonetheless, a ``EmbFile`` keeps track of all its open readers and close them
    automatically when it is closed.

    Args:
        out_dtype:
            all the vectors will be converted to this dtype before being returned

    Attributes:
        out_dtype (numpy.dtype):
            all the vectors will be converted to this data type before being returned
    """

    def __init__(self, out_dtype: DType):
        self.out_dtype = numpy.dtype(out_dtype)
        self._closed = False
        self._current_vector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        return self.next_word()

    def _raise_illegal_operation(self, *args, **kwargs):
        raise IllegalOperation('attempted to use a closed reader')

    # noinspection PyAttributeOutsideInit
    def close(self) -> None:
        """ Closes the reader """
        if self._closed:
            return
        self._close()
        self._closed = True
        self.reset = self.next_word = self._raise_illegal_operation  # type: ignore

    @abc.abstractmethod
    def _close(self) -> None:
        """ *(Abstract method)* Closes the reader """

    @abc.abstractmethod
    def reset(self) -> None:
        """ *(Abstract method)* Brings back the reader to the first word vector pair """

    @abc.abstractmethod
    def next_word(self) -> str:
        """ *(Abstract method)* Reads and returns the next word in the file. """

    @abc.abstractmethod
    def current_vector(self) -> VectorType:
        """ *(Abstract method)* The vector for the current word (i.e. the last word read).
        If accessed before any word has been read, it raises ``IllegalOperation``.
        The dtype of the returned vector is cls.out_dtype. """


class AbstractEmbFileReader(EmbFileReader, abc.ABC):
    """
    *(Abstract class)* Facilitates the implementation of a :class:`EmbFileReader`, especially for a
    file that stores a word and its vector nearby in the file (txt and bin formats), though it can
    be used for other kind of formats as well if it looks convenient. It:

    - keeps track of whether the reader is pointing to a word or a vector and skips the vector
      when it is not requested during an iteration
    - caches the current vector once it is read

    Sub-classes must implement:

    .. autosummary::
        _read_word
        _read_vector
        _skip_vector
        _close

    .. automethod:: _read_word
    .. automethod:: _read_vector
    .. automethod:: _skip_vector
    .. automethod:: _reset
    .. automethod:: _close
    """

    def __init__(self, out_dtype: DType):
        super().__init__(out_dtype)
        self._closed = False
        self._pointing_to_vector = False  # True if the file reader is pointing to vector data
        self._current_vector = None

    @abc.abstractmethod
    def _reset(self) -> None:
        """ *(Abstract method)* Resets the reader """

    @overrides
    def reset(self) -> None:  # type: ignore
        """ Brings back the reader to the beginning of the file """
        self._reset()
        self._pointing_to_vector = False
        self._current_vector = None

    @abc.abstractmethod
    def _read_word(self) -> str:
        """
        *(Abstract method)* Reads a word assuming the next thing to read in the file is a word.
        It must raise StopIteration if there's not another word to read.
        """

    @abc.abstractmethod
    def _read_vector(self) -> VectorType:
        """
        *(Abstract method)* Reads the vector for the last word read. This method is never called if
        no word has been read or at the end of file. It is called at most time per word.
        """

    @abc.abstractmethod
    def _skip_vector(self) -> None:
        """
        *(Abstract method)* Called when we want to read the next word without loading
        the vector for the current word. For some formats, it may be empty.
        """

    @overrides
    def next_word(self) -> str:  # type: ignore
        """ Reads and returns the next word in the file. """
        if self._pointing_to_vector:
            self._skip_vector()
            self._pointing_to_vector = False

        word = self._read_word()  # this can raise StopIteration
        self._pointing_to_vector = True
        self._current_vector = None
        return word

    @property
    def current_vector(self) -> VectorType:
        """
        The vector associated to the current word (i.e. the last word read).
        If accessed before any word has been read, it raises ``IllegalOperation``.
        """
        if self._current_vector is None:
            if not self._pointing_to_vector:
                raise IllegalOperation('you called current_vector before reading any word')
            self._current_vector = self._read_vector()
            self._pointing_to_vector = False
        return self._current_vector
