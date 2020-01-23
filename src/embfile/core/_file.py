__all__ = ['EmbFile']

import abc
import itertools
import warnings
from pathlib import Path
from typing import (Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple,
                    TypeVar, Union)

import numpy

from embfile._utils import coalesce, maybe_progbar, noop
from embfile.compression import COMPRESSION_TO_EXTENSIONS, remove_compression_extension
from embfile.core.loaders import SequentialLoader, VectorsLoader
from embfile.core.reader import EmbFileReader
from embfile.errors import IllegalOperation
from embfile.types import DType, PairsType, PathType, VectorType
from embfile.word_vector import WordVector

#: Default verbosity mode
DEFAULT_VERBOSE: bool = True


class EmbFile(abc.ABC):
    """
    *(Abstract class)* The base class of all the embedding files.

    Sub-classes must:

    #. ensure they set attributes :attr:`.vocab_size` and :attr:`.vector_size` when a file
       instance is created
    #. implement a :class:`~embfile.core.EmbFileReader` for the format and implements
       the abstract method :meth:`_reader`
    #. implement the abstract method :meth:`_close`
    #. *(optionally)* implement a :class:`~embfile.core.loaders.VectorsLoader` (if they can improve
       upon the default loader) and override :meth:`loader`
    #. *(optionally)* implement a :class:`~embfile.core.EmbFileCreator` for the format and set
       the class constant :attr:`.Creator`

    Args:
        path (Path):
            path of the embedding file (eventually compressed)

        out_dtype (numpy.dtype):
            all the vectors will be converted to this data type. The sub-class
            is responsible to set a suitable default value.

        verbose (bool):
            whether to show a progress bar by default in all time-consuming operations

    Attributes:
        path (Path):
            path of the embedding file

        vocab_size (int or ``None``):
            number of words in the file (can be ``None`` for some ``TextEmbFile``)

        vector_size (int):
            length of the vectors

        verbose (bool):
            whether to show a progress bar by default in all time-consuming operations

        closed (bool):
            True if the file was closed

    .. automethod:: _reader
    .. automethod:: _close
    """
    DEFAULT_EXTENSION: str

    def __init__(self, path: PathType,
                 out_dtype: Optional[DType] = None,
                 verbose: bool = DEFAULT_VERBOSE):

        path = Path(path)
        self.path = path
        self.out_dtype = numpy.dtype(out_dtype)
        self.verbose = verbose

        self.vocab_size: Optional[int] = None
        self.vector_size = -1

        self._objects_to_close: List[Union[EmbFileReader, VectorsLoader]] = list()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        classname = self.__class__.__name__
        return '{} (\n' \
               '  path = {},\n' \
               '  vocab_size = {},\n' \
               '  vector_size = {}\n' \
               ')'.format(classname, self.path, self.vocab_size, self.vector_size)

    @abc.abstractmethod
    def _close(self) -> None:
        """ *(Abstract method)* Releases eventual resources used by the EmbFile. """

    def close(self) -> None:
        """ Releases all the open resources linked to this file, including the opened readers. """
        for obj in self._objects_to_close:
            obj.close()
        self._objects_to_close.clear()
        self._close()
        self.closed = True

    @abc.abstractmethod
    def _reader(self) -> EmbFileReader:
        """ *(Abstract method)* Returns a new reader for the file which allows to iterate
        efficiently the word-vectors inside it. Called by :meth:`reader`. """

    def reader(self) -> EmbFileReader:
        """ Creates and returns a new file reader. When the file is closed, all the still opened
        readers are closed automatically. """
        if self.closed:
            raise IllegalOperation('attempted to use a closed file')
        reader = self._reader()
        self._objects_to_close.append(reader)
        return reader

    def _loader(self, words: Iterable[str], missing_ok: bool = True,
                verbose: Optional[bool] = None) -> 'VectorsLoader':
        return SequentialLoader(self, words, missing_ok)

    def loader(self, words: Iterable[str], missing_ok: bool = True,
               verbose: Optional[bool] = None) -> 'VectorsLoader':
        """
        Returns a :class:`~embfile.core.loaders.VectorsLoader`, an iterator that looks for the
        provided words in the file and yields available (word, vector) pairs one by one.
        If ``missing_ok=True`` (default), provides the set of missing words in the
        property ``missing_words`` (once the iteration ends).

        See :class:`embfile.core.VectorsLoader` for more info.

        Example:
            You should use a loader when you need to load many vectors in some custom data structure
            and you don't want to waste memory (e.g. build_matrix uses it to load the vectors
            directly into the matrix)::

                data_structure = MyCustomStructure()
                with file.loader(many_words) as loader:
                    for word, vector in loader:
                        data_structure[word] = vector
                print('Number of missing words:', len(loader.missing_words)

        See Also:
            :meth:`load`
            :meth:`find`
        """
        if self.closed:
            raise IllegalOperation('attempted to use a closed file')
        loader = self._loader(words, missing_ok, verbose=verbose)
        self._objects_to_close.append(loader)
        return loader

    def words(self) -> Iterable[str]:
        """ Returns an iterable for all the words in the file. """
        with self.reader() as reader:
            yield from reader

    def vectors(self) -> Iterable[VectorType]:
        """ Returns an iterable for all the vectors in the file. """
        with self.reader() as reader:
            for _ in reader:
                yield reader.current_vector

    def word_vectors(self) -> Iterable[WordVector]:
        """ Returns an iterable for all the (word, vector) word_vectors in the file. """
        with self.reader() as reader:
            for word in reader:
                yield WordVector(word, reader.current_vector)

    def _maybe_progbar(self, iterable: Iterable, enable: bool = None, **kwargs):
        return maybe_progbar(iterable, yes=coalesce(enable, self.verbose), **kwargs)

    def to_dict(self, verbose: Optional[bool] = None) -> Dict[str, VectorType]:
        """ Returns the entire file content in a dictionary word -> vector. """
        word_vectors = self._maybe_progbar(self.word_vectors(), verbose,
                                           total=self.vocab_size, desc='Loading to dict')
        return dict(word_vectors)

    def to_list(self, verbose: Optional[bool] = None) -> List[WordVector]:
        """ Returns the entire file content in a list of :class:`WordVector`'s. """
        word_vectors = self._maybe_progbar(self.word_vectors(), verbose,
                                           total=self.vocab_size, desc='Loading to list')
        return list(word_vectors)

    def load(self, words: Iterable[str], verbose: Optional[bool] = None) -> Dict[str, VectorType]:
        """
        Loads the vectors for the input words in a ``{word: vec}`` dict, raising
        ``KeyError`` if any word is missing.

        Args:
            words:
                the words to get
            verbose:
                if None, self.verbose is used

        Returns:
            (Dict[str, VectorType]): a dictionary ``{word: vector}``

        See Also:
            :meth:`find` - it returns the set of all missing words, instead of raising
            ``KeyError``.
        """
        with self.loader(words, missing_ok=False, verbose=verbose) as loader:
            return {word: vector for word, vector in loader}

    class _FindOutput(NamedTuple):
        word2vec: Dict[str, VectorType]
        missing_words: Set[str]

    def find(self, words: Iterable[str],
             verbose: Optional[bool] = None) -> _FindOutput:  # noqa: F821
        """
        Looks for the input words in the file, return: 1) a dict ``{word: vec}``
        containing the available words and 2) a set containing the words not found.

        Args:
            words (Iterable[str]):
                the words to look for
            verbose:
                if None, self.verbose is used

        Returns:
            namedtuple:
                a namedtuple with the following fields:

                - **word2vec** (*Dict[str, VectorType]*): dictionary ``{word: vector}``
                - **missing_words** (*Set[str]*): set of words not found in the file

        See also:
            :meth:`load` - which raises KeyError if any word is not found in the file.
        """
        with self.loader(words, verbose=verbose) as loader:
            word2vec = {word: vec for word, vec in loader}
            return EmbFile._FindOutput(word2vec=word2vec,  # noqa
                                       missing_words=loader.missing_words)

    def filter(self, condition: Callable[[str], bool],
               verbose: Optional[bool] = None) -> Iterator[Tuple[str, VectorType]]:
        """
        Returns a generator that yields a word vector pair for each word in the file that satisfies
        a given condition. For example, to get all the words starting with "z"::

            list(file.filter(lambda word: word.startswith('z')))

        Args:
            condition:
                a function that, given a word in input, outputs True if the word should be taken
            verbose:
                if True, a progress bar is showed (the bar is updated each time a word is read, not
                each time a word vector pair is yielded).
        """
        with self.reader() as reader:
            for word in self._maybe_progbar(reader, verbose, total=self.vocab_size,
                                            desc="Filtering word vectors"):
                if condition(word):
                    yield word, reader.current_vector

    def save_vocab(self, path: PathType = None,
                   encoding: str = 'utf-8',
                   overwrite: bool = False,
                   verbose: Optional[bool] = None) -> Path:
        """
        Save the vocabulary of the embedding file on a text file. By default the file is saved
        in the same directory of the embedding file, e.g.::

            /path/to/filename.txt.gz  ==> /path/to/filename_vocab.txt

        Args:
            path:
                where to save the file
            encoding:
                text encoding
            overwrite:
                if the file exists and it is True, overwrite the file
            verbose:
                if None, self.verbose is used

        Returns:
            (Path): the path to the vocabulary file
        """
        if path is None:
            basename = self.path.name.split('.')[0]
            filename = basename + '_vocab.txt'
            path = self.path.parent / filename

        if path.exists() and not overwrite:
            raise FileExistsError(path)

        with open(path, 'wt', encoding=encoding) as vocab_file:
            for word in self._maybe_progbar(self.words(), verbose, total=self.vocab_size,
                                            desc='Saving vocabulary'):
                vocab_file.write(word + '\n')

        return path

    @classmethod
    @abc.abstractmethod
    def _create(cls, out_path: Path,
                word_vectors: Iterable[Tuple[str, VectorType]],
                vector_size: int,
                vocab_size: Optional[int],
                compression: Optional[str] = None,
                verbose: bool = True, **format_kwargs) -> None:
        """
        The core method that actually writes word vectors to disk and it's format-specific.

        This method is called by the public ``create`` method after it performs boring
        args checking and normalization.

        Note that ``vocab_size`` can be ``None``: it is up to the specific implementation
        to treat it as an error or not.

        For implementors:
        #. replace the generic ``**format_kwargs`` with format-specific arguments
        #. you can safely assume ``out_path.exists() is False``
        #. you should warn the user if the provided ``vocab_size`` is not equal to the
           actual number of written word vectors (use :func:`warn_if_wrong_vocab_size`)
        #. should raise ValueError if a vector has a different size than expected
           (use :func:`check_vector_size`)
        """

    @classmethod
    def create(cls, out_path: PathType,
               word_vectors: PairsType,
               vocab_size: Optional[int] = None,
               compression: Optional[str] = None,
               verbose: bool = True, overwrite: bool = False,
               **format_kwargs) -> None:
        """
        Creates a file on disk containing the provided word vectors.

        Args:
            out_path:
                path to the created file

            word_vectors (Dict[str, VectorType] or Iterable[Tuple[str, VectorType]]):
                it can be an iterable of word vector tuples or a dictionary ``word -> vector``;
                the word vectors are written in the order determined by the iterable object.

            vocab_size:
                it must be provided if ``word_vectors`` has no ``__len__`` and the specific-format
                creator needs to know a priori the vocabulary size; in any case, the creator
                should check at the end that the provided ``vocab_size`` matches the actual length
                of ``word_vectors``

            compression:
                valid values are: ``"bz2"|"bz", "gzip"|"gz", "xz"|"lzma", "zip"``

            verbose:
                if positive, show progress bars and information

            overwrite:
                overwrite the file if it already exists

            format_kwargs:
                format-specific arguments
        """
        echo = print if verbose else noop
        out_path = Path(out_path)
        if out_path.exists():
            if overwrite:
                out_path.unlink()
                echo('the file %s already exists and overwriting is enabled, '
                     'so it was removed' % out_path)
            else:
                raise FileExistsError('the file %s already exists' % out_path)

        # if ``word_vectors`` has length, we use that as ``vocab_size``
        try:
            actual_vocab_size = len(word_vectors)
        except TypeError:
            pass
        else:
            if vocab_size and vocab_size != actual_vocab_size:
                warnings.warn('you provided vocab_size=%d but the actual vocab_size is %d; we will '
                              'use the actual vocab_size' % (vocab_size, actual_vocab_size))
            vocab_size = actual_vocab_size

        pairs: Iterable[Tuple[str, VectorType]]  # for mypy
        if isinstance(word_vectors, dict):
            pairs = word_vectors.items()
        elif isinstance(word_vectors, Iterable):
            pairs = word_vectors
        else:
            raise TypeError('word_vectors is neither a dict nor an iterable: %r' % word_vectors)

        # To get [vector_size] we have to "glance" at the first vector in [pairs];
        # in case pairs is an iterator, [glance_first_element] returns a new iterable
        # itertools.
        (_, first_vector), pairs = glance_first_element(pairs)
        vector_size = len(first_vector)

        cls._create(out_path, pairs,
                    vector_size=vector_size, vocab_size=vocab_size,
                    compression=compression, verbose=verbose, **format_kwargs)

        echo('Creation completed: %s' % out_path)

    @classmethod
    def create_from_file(cls, source_file: 'EmbFile',
                         out_dir: Optional[PathType] = None,
                         out_filename: Optional[str] = None,
                         vocab_size: Optional[int] = None,
                         compression: Optional[str] = None,
                         verbose: bool = True,
                         overwrite: bool = False,
                         **format_kwargs) -> Path:
        """
        Creates a new file on disk with the same content of another file.

        Args:
            source_file:
                the file to take data from

            out_dir:
                directory where the file will be stored; by default, it's the parent directory
                of the source file

            out_filename:
                filename of the produced name (inside ``out_dir``); by default, it is obtained by
                replacing the extension of the source file with the proper one and appending the
                compression extension if ``compression is not None``.
                **Note:** if you pass this argument, the compression extension is not automatically
                appended.

            vocab_size:
                if the source EmbFile has attribute ``vocab_size == None``, then: if the specific
                creator requires it (bin and txt formats do), it `must` be provided; otherwise it
                `can` be provided for having ETA in progress bars.

            compression:
                valid values are: ``"bz2"|"bz", "gzip"|"gz", "xz"|"lzma", "zip"``

            verbose:
                print info and progress bar

            overwrite:
                overwrite a file with the same name if it already exists

            format_kwargs:
                format-specific arguments (see above)

        """
        vocab_size = source_file.vocab_size or vocab_size

        out_dir = Path(out_dir or source_file.path.parent)

        if out_filename is None:
            suffix = cls.DEFAULT_EXTENSION
            if compression:
                suffix += COMPRESSION_TO_EXTENSIONS[compression][0]
            source_path = remove_compression_extension(source_file.path)
            out_filename = source_path.with_suffix(suffix).name

        out_path = out_dir / out_filename

        cls.create(out_path, word_vectors=source_file.word_vectors(),
                   vocab_size=vocab_size, compression=compression,
                   verbose=verbose, overwrite=overwrite, **format_kwargs)

        return out_path


# ==============================
#  Utility functions
# ==============================
T = TypeVar('T')


def glance_first_element(iterable: Iterable[T]) -> Tuple[T, Iterable[T]]:
    iterator = iter(iterable)
    first = next(iterator)
    if iterable is iterator:  # iterable is an iterator
        # methaphorically put the first element "back in"
        return first, itertools.chain([first], iterable)
    else:
        return first, iterable


def warn_if_wrong_vocab_size(expected_size, actual_size, extra_info=''):
    if actual_size is not None and actual_size != expected_size:
        fmt = ('the actual number of word vectors in the iterator/file was different than the '
               'provided/expected one; expected: %d; actual: %d.\n' + extra_info)
        warnings.warn(fmt % (expected_size, actual_size))


def check_vector_size(i, vector, vector_size):
    if len(vector) != vector_size:
        raise ValueError('inconsistent vector_size: the first vector has size %d but the vector of '
                         'index %d has size %d' % (vector_size, i, len(vector)))
