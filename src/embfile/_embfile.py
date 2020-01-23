from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Type, Union

import numpy

from embfile._utils import coalesce, maybe_progbar, noop, require
from embfile.compression import EXTENSION_TO_COMPRESSION
from embfile.core import EmbFile
from embfile.formats import BinaryEmbFile, TextEmbFile, VVMEmbFile
from embfile.initializers import Initializer, NormalInitializer
from embfile.registry import FormatsRegistry
from embfile.types import DType, PathType, VectorType

#: Store the mapping between each EmbFile concrete class and both its format_id and
#: its associated extensions.
FORMATS = FormatsRegistry()

FORMATS.register_format(BinaryEmbFile, format_id='bin', extensions=['.bin'])
FORMATS.register_format(TextEmbFile, format_id='txt', extensions=['.txt', '.vec'])
FORMATS.register_format(VVMEmbFile, format_id='vvm', extensions=['.vvm'])


def register_format(format_id: str, extensions: Iterable[str], overwrite: bool = False):
    """
    Class decorator that associates a new ``EmbFile`` sub-class with a format_id and one or
    multiple extensions. Once you register a format, you can use :func:`~embfile.open` to open
    files of that format.
    """

    def decorator(embfile_class: Type['EmbFile']):
        FORMATS.register_format(embfile_class, format_id, extensions, overwrite)
        return embfile_class

    return decorator


def associate_extension(ext: str, format_id: str, overwrite: bool = False):
    """ Associates a file extension to a registered embedding file format. """
    FORMATS.associate_extension(ext, format_id, overwrite)


def _infer_format_from_path(path: PathType) -> Type[EmbFile]:
    # Ignore the file extension due to compression (.txt.gz --> txt)
    path = Path(path)
    suffixes = path.suffixes
    if not suffixes:
        format_extension = None
    elif suffixes[-1] in EXTENSION_TO_COMPRESSION:
        format_extension = suffixes[-2] if len(suffixes) > 1 else None
    else:
        format_extension = suffixes[-1]

    if not format_extension:
        raise ValueError(
            'unable to infer the file type because the file has no extension: ' + path.name)

    return _extension_to_class(format_extension)


def _extension_to_class(format_extension: str) -> Type[EmbFile]:
    try:
        return FORMATS.extension_to_class(format_extension)
    except KeyError:
        raise ValueError(
            'unknown file extension "{ext}";\n'
            'supported extensions/formats are: {extensions}.\n'
            'If your file is supported but has just an unknown extension, you can alternatively:\n'
            '- pass the argument format_id \n'
            '- or associate the extension to one of the formats with embfile.associate_extension()'
            .format(ext=format_extension, extensions=', '.join(FORMATS.extensions())))


def open(path: PathType,
         format_id: Optional[str] = None,
         **format_kwargs) -> EmbFile:
    """
    Opens an embedding file inferring the file format from the file extension (if not explicitly
    provided in ``format_id``). Note that you can always open a file using the specific ``EmbFile``
    subclass; it can be more convenient since you get auto-completion and quick doc for
    format-specific arguments.

    Example::

        with embfile.open('path/to/embfile.txt') as f:
            # do something with f

    Supported formats:

    +---------------------------------+-----------+------------+-----------------------------------+
    | Class                           | format_id | Extensions | Description                       |
    +=================================+===========+============+===================================+
    | :class:`~embfile.TextEmbFile`   | txt       | .txt, .vec |  Glove/fastText format            |
    +---------------------------------+-----------+------------+-----------------------------------+
    | :class:`~embfile.BinaryEmbFile` | bin       | .bin       |  Google word2vec format           |
    +---------------------------------+-----------+------------+-----------------------------------+
    | :class:`~embfile.VVMEmbFile`    | vvm       | .vvm       | A tarball containing three files: |
    |                                 |           |            | vocab.txt, vectors.bin, meta.json |
    +---------------------------------+-----------+------------+-----------------------------------+

    You can **register new formats or extensions** using the functions
    :func:`embfile.register_format` and :func:`embfile.associate_extension`.

    Args:
        path:
            path to the file

        format_id:
            string ID of the embedding file format. If not provided, it is inferred from the
            file name. Valid choices are: 'txt', 'bin', 'vvm'.

        format_kwargs:
            additional format-specific arguments (see doc for specific file formats)

    Returns:
        An instance of a concrete subclass of :class:`~embfile.core.EmbFile` .

    See also:
        :func:`embfile.register_format`:
            registers your custom EmbFile implementation so it is recognized by this function
        :func:`embfile.associate_extension`:
            associates an extension to a registered format
    """
    if format_id is None:
        format_class = _infer_format_from_path(path)
    else:
        try:
            format_class = FORMATS.id_to_class[format_id]
        except KeyError:
            raise ValueError('Unknown format: "{}".\nKnown formats are: {}'
                             .format(format_id, list(FORMATS.format_ids())))

    return format_class(path, **format_kwargs)


class BuildMatrixOutput(NamedTuple):
    """ NamedTuple returned by :meth:`build_matrix` """
    matrix: numpy.ndarray
    word2index: Dict[str, int]
    missing_words: Set[str]

    def found_words(self):
        return set(self.word2index) - self.missing_words

    def word_indexes(self, words) -> List[int]:
        return [self.word2index[w] for w in words]

    def vector(self, word):
        return self.matrix[self.word2index[word]]

    def pretty(self, precision=3, threshold=5):
        """ Pretty string method for documentation purposes. """
        rows = [numpy.array2string(row, precision=precision, threshold=threshold,
                                   sign=' ', floatmode='fixed')
                for row in self.matrix]
        index_width = len(str(len(rows) - 1))
        annotations = ['  # {:>{w}}: '.format(i, w=index_width) for i in range(len(rows))]
        for word, index in self.word2index.items():
            annotations[index] += word
            if word in self.missing_words:
                annotations[index] += ' [out of file vocabulary]'
        return '\n'.join(row + annotation for row, annotation in zip(rows, annotations))


def build_matrix(f: EmbFile, words: Union[Iterable[str], Dict[str, int]],
                 start_index: int = 0,
                 dtype: Optional[DType] = None,
                 oov_initializer: Optional[Callable[[Sequence[int]], VectorType]]
                 = NormalInitializer(),
                 verbose: Optional[bool] = None) -> BuildMatrixOutput:  # noqa: F821
    """
    Creates an embedding matrix for the provided words. ``words`` can be:

    1. an **iterable of strings** -- in this case, the words in the iterable are mapped
       to consecutive rows of the matrix starting from the row of index ``start_index``
       (by default, ``0``); the rows with index ``i < start_index`` are left to zeros.

    2. a **dictionary** ``{word -> index}`` that maps each word to a row --
       in this case, the matrix has shape::

            [max_index + 1, vector_size]

       where ``max_index = max(word_to_index.values())``. The rows that are not associated
       with any word are left to zeros. If multiple words are mapped to the same row, the
       function raises ``ValueError``.

    In both cases, all the word vectors that are not found in the file are initialized using
    ``oov_initializer``, which can be:

    1. ``None`` -- leave missing vectors to zeros

    2. a function that takes the shape of the array to generate (a tuple) as first argument::

        oov_initializer=lambda shape: numpy.random.normal(scale=0.01, size=shape)
        oov_initializer=numpy.ones  # don't use this for word vectors :|

    3. an instance of :class:`~embfile.initializers.Initializer`, which is a "fittable"
       initializer; in this case, the initializer is fit on the found vectors (the vectors that
       are both in ``vocab`` and in the file).

    By default, `oov_initializer` is an instance of
    :class:`~embfile.initializers.NormalInitializer`
    which generates vectors using a normal distribution with the same mean and standard
    deviation of the vectors found.

    Args:
        f:
            the file containing the word vectors

        words (Iterable[str] or Dict[str, int]):
            iterable of words or dictionary that maps each word to a row index

        start_index (int):
            ignored if ``vocab`` is a dict; if ``vocab`` is a collection, determines the index
            associated to the first word (and so, the number of rows left to zeros at the
            beginning of the matrix)

        dtype (optional, DType):
            matrix data type; if ``None``, ``cls.out_dtype`` is used

        oov_initializer (optional, Callable or :class:`~embfile.initializers.Initializer`):
            initializer for out-of-(file)-vocabulary word vectors. See the class docstring for
            more information.

        verbose (bool):
            if None, f.verbose is used
    """
    start_time = timer()

    dtype = numpy.dtype(dtype) if dtype is not None else f.out_dtype
    require(words, 'empty vocab argument')
    require(start_index >= 0, 'negative start_index')
    echo = print if coalesce(verbose, f.verbose) else noop

    if isinstance(words, dict):
        word2index = words
        num_rows = 1 + max(word2index.values())
        if len(set(word2index.values())) < len(word2index):
            raise ValueError('multiple words are mapped to the same row')
    elif isinstance(words, Iterable):
        word2index = {word: index for index, word in enumerate(words, start_index)}
        num_rows = start_index + len(word2index)
    else:
        raise TypeError('vocab must be of type Iterable[str] or a Dict[str, int]; '
                        'it is: %r' % words)

    matrix = numpy.zeros((num_rows, f.vector_size), dtype=dtype)

    found_word_indexes = []
    with f.loader(word2index, verbose=verbose) as loader:
        for word, vector in loader:
            idx = word2index[word]
            matrix[idx] = vector
            found_word_indexes.append(idx)
    missing_words = loader.missing_words

    # Initialize vectors of out-of-file-vocabulary words
    if missing_words and oov_initializer:
        if isinstance(oov_initializer, Initializer) and found_word_indexes:
            echo('Fitting the vector initializer')
            found_vectors = matrix[found_word_indexes]
            oov_initializer.fit(found_vectors)

        # @MAYBE: replace this loop with oov_initializer((len(missing_words), vector_size)
        missing_indexes = [word2index[word] for word in missing_words]
        vector_shape = (f.vector_size,)
        desc = 'Initializing missing word vectors'
        for idx in maybe_progbar(missing_indexes, verbose, desc=desc):
            matrix[idx] = oov_initializer(vector_shape)

    elapsed_time = timer() - start_time
    echo('Total elapsed time for creating the matrix: %.1f seconds' % elapsed_time)
    return BuildMatrixOutput(matrix, word2index, missing_words)
