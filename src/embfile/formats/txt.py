__all__ = ['TextEmbFile', 'TextEmbFileReader']

import codecs
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TextIO, Tuple

import numpy
from overrides import overrides

from embfile._utils import progbar
from embfile.compression import open_file
from embfile.core import AbstractEmbFileReader, EmbFile
from embfile.core._file import DEFAULT_VERBOSE, check_vector_size, warn_if_wrong_vocab_size
from embfile.errors import BadEmbFile
from embfile.types import DType, PairsType, PathType, VectorType

#: Default text encoding
DEFAULT_ENCODING = 'utf-8'

#: Default vector data type (little-endian single-precision floating point numbers)
DEFAULT_OUT_DTYPE = numpy.dtype('<f4')


class TextEmbFileReader(AbstractEmbFileReader):
    """ :class:`~embfile.core.EmbFileReader` for the textual format. """

    @classmethod
    def from_path(cls, path: PathType,
                  encoding: str = DEFAULT_ENCODING,
                  out_dtype: DType = DEFAULT_OUT_DTYPE,
                  vocab_size: Optional[int] = None) -> 'TextEmbFileReader':
        """
        Returns a `TextEmbFileReader` from the path of a (eventually compressed) text file.
        """
        return cls(open_file(path, 'rt', encoding=encoding),
                   out_dtype=out_dtype, vocab_size=vocab_size)

    def __init__(self, file_obj: TextIO,
                 out_dtype: DType = DEFAULT_OUT_DTYPE,
                 vocab_size: Optional[int] = None) -> None:

        super().__init__(out_dtype)

        self._file = file_obj
        self.out_dtype = numpy.dtype(out_dtype)

        first_line = file_obj.readline()
        self.header = header = self.parse_header(first_line)

        if header:
            self.vocab_size = header['vocab_size']
            self.vector_size = header['vector_size']
        else:
            self.vocab_size = vocab_size
            self.vector_size = len(first_line.split(' ')) - 1
            file_obj.seek(0)

        self._current_line: Optional[str] = None
        self._vector_start: int

    @classmethod
    def parse_header(cls, line: str) -> Dict[str, Any]:
        fields = line.split(' ')
        if len(fields) != 2:
            return dict()
        try:
            vocab_size, vector_size = map(int, fields)
        except ValueError:
            return dict()
        else:
            return {'vocab_size': vocab_size, 'vector_size': vector_size}

    @overrides
    def _close(self) -> None:
        self._file.close()

    @overrides
    def _reset(self) -> None:
        self._file.seek(0)
        if self.header:
            self._file.readline()  # skip the header

    @overrides
    def _read_word(self) -> str:
        self._current_line = line = next(self._file)
        word_end = line.find(' ')
        if word_end < 0:
            raise BadEmbFile(
                'did not find any space in the entire line: {!r}'.format(line))
        self._vector_start = word_end + 1
        return line[:word_end]

    @overrides
    def _read_vector(self) -> VectorType:
        line = self._current_line
        # `line` is surely not None because AbstractEmbFileReader, by contract, never calls
        # _read_vector() neither before any word is read nor at the end of the file.
        vector = numpy.fromstring(line[self._vector_start:], sep=' ')  # type: ignore
        if self.vector_size != len(vector):
            raise BadEmbFile(
                'vector_size is %d but the following line contains %d elements after the '
                'word: "%s"\n The file may be corrupted.' % (self.vector_size, len(vector), line))
        return numpy.asarray(vector, dtype=self.out_dtype)

    @overrides
    def _skip_vector(self) -> None:
        pass


class TextEmbFile(EmbFile):
    """
    The format used by Glove and FastText files. Each vector pair is stored as a line of text made
    of space-separated fields::

        word vec[0] vec[1] ... vec[vector_size-1]

    It may have or not an (automatically detected) "header", containing ``vocab_size`` and
    ``vector_size`` (in this order).

    If the file doesn't have a header, ``vector_size`` is set to the length of the first vector.
    If you know ``vocab_size`` (even an approximate value), you may want to provide it to have ETA
    in progress bars.

    If the file has a header and you provide ``vocab_size``, the provided value is ignored.

    Compressed files are decompressed while you proceed reeding. Note that each file reader
    will decompress the file independently, so if you need to read the file multiple times
    it's better you decompress the entire file first and then open it.

    Attributes:
        path
        encoding
        out_dtype
        verbose

    """
    DEFAULT_EXTENSION = '.txt'

    def __init__(self, path: PathType, encoding: str = DEFAULT_ENCODING,
                 out_dtype: DType = 'float32',
                 vocab_size: Optional[int] = None,
                 verbose: int = DEFAULT_VERBOSE):
        """
        Args:
            path:
                path to the embedding file
            encoding:
                encoding of the text file; default is utf-8
            out_dtype:
                the dtype of the vectors that will be returned; default is single-precision float
            vocab_size:
                useful when the file has no header but you know vocab_size;
                if the file has a header, this argument is ignored.
            verbose:
                default level of verbosity for all methods
        """
        super().__init__(path, out_dtype=out_dtype, verbose=verbose)

        self.encoding = encoding
        self.vocab_size = vocab_size
        with self.reader() as reader:
            self.vocab_size = reader.vocab_size
            self.vector_size = reader.vector_size

    @overrides
    def _reader(self) -> 'TextEmbFileReader':
        return TextEmbFileReader.from_path(
            path=self.path,
            encoding=self.encoding,
            out_dtype=self.out_dtype,
            vocab_size=self.vocab_size
        )

    @overrides
    def _close(self) -> None:
        pass

    @classmethod
    def _create(cls, out_path: Path, word_vectors: Iterable[Tuple[str, VectorType]],
                vector_size: int, vocab_size: int,
                compression: Optional[str] = None,
                verbose: bool = True,
                encoding: str = DEFAULT_ENCODING,
                precision: int = 5) -> Path:

        number_fmt = '%.{}f'.format(precision)

        if not vocab_size:
            raise ValueError('unable to infer vocab_size; you must manually provide it')

        # Because of a bug that io.TextIOWrapper presents when used in combination with bz2 and lzma
        # we have to complicate things a little bit. For more info about the bug (that I discovered
        # testing this code) see:
        # https://stackoverflow.com/questions/55171439/python-bz2-and-lzma-in-mode-wt-dont-write-the-bom-while-gzip-does-why);
        # I'll open the file in binary mode and encode the text using an IncrementalEncoder for
        # writing the BOM only at the beginning.
        encoder = codecs.getincrementalencoder(encoding)()
        encode = encoder.encode
        with open_file(out_path, 'wb', compression=compression) as fout:

            fout.write(encode('%d %d\n' % (vocab_size, vector_size)))

            for i, (word, vector) in progbar(enumerate(word_vectors), enable=verbose,
                                             desc='Writing',
                                             total=vocab_size):
                if ' ' in word:
                    raise ValueError("the word number %d contains one or more spaces: %r"
                                     % (i, word))
                check_vector_size(i, vector, vector_size)

                fout.write(encode(word + ' '))
                vector_string = ' '.join(number_fmt % num for num in vector)
                fout.write(encode(vector_string))
                fout.write(encode('\n'))

        warn_if_wrong_vocab_size(vocab_size, actual_size=i + 1,
                                 extra_info='As a consequence, the header of the file has a wrong '
                                            'vocab_size. You can change it editing the file.')

        return out_path

    # For documentation/auto-completion purposes, let's add format-specific args in the signature
    @classmethod
    def create(cls, out_path: PathType, word_vectors: PairsType, vocab_size: Optional[int] = None,
               compression: Optional[str] = None, verbose: bool = True, overwrite: bool = False,
               encoding: str = DEFAULT_ENCODING,
               precision: int = 5) -> None:
        super().create(out_path, word_vectors, vocab_size, compression, verbose, overwrite,
                       encoding=encoding, precision=precision)

    @classmethod
    def create_from_file(cls, source_file: 'EmbFile', out_dir: Optional[PathType] = None,
                         out_filename: Optional[str] = None, vocab_size: Optional[int] = None,
                         compression: Optional[str] = None, verbose: bool = True,
                         overwrite: bool = False, encoding: str = DEFAULT_ENCODING,
                         precision: int = 5) -> Path:
        return super().create_from_file(source_file, out_dir, out_filename, vocab_size, compression,
                                        verbose, overwrite, encoding=encoding, precision=precision)
