__all__ = ['VVMEmbFile', 'VVMEmbFileReader']

import io
import json
import os
import shutil
import tarfile
import tempfile
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import IO, Iterable, Optional, Tuple

import numpy
from overrides import overrides

from embfile._utils import MappingComposition, coalesce, maybe_progbar, noop, progbar
from embfile.compression import open_file
from embfile.core._file import (DEFAULT_VERBOSE, EmbFile, check_vector_size, glance_first_element,
                                warn_if_wrong_vocab_size)
from embfile.core.loaders import RandomAccessLoader, Word2Vector
from embfile.core.reader import AbstractEmbFileReader
from embfile.types import DType, PairsType, PathType, VectorType

VOCAB_FILENAME = 'vocab.txt'
VECTORS_FILENAME = 'vectors.bin'
META_FILENAME = 'meta.json'

_TAR_COMPRESSIONS = {'gz', 'bz2', 'xz'}
DEFAULT_EXTENSION = '.vvm'


#: Default text encoding
DEFAULT_ENCODING = 'utf-8'

#: Default vector data type (little-endian single-precision floating point numbers)
DEFAULT_DTYPE = numpy.dtype('<f4')


class VVMEmbFileReader(AbstractEmbFileReader):
    """ :class:`~embfile.core.EmbFileReader` for the vvm format. """

    def __init__(self, file, vectors_file) -> None:
        super().__init__(file.out_dtype)
        self.file = file
        self.vector_size = file.vector_size
        self.dtype = file.dtype
        self._vectors_file = vectors_file
        self._vector_size_in_bytes = file.vector_size * file.dtype.itemsize
        self._words_iter = iter(file.vocab)

    def _close(self) -> None:
        self._vectors_file.close()

    def _reset(self) -> None:
        self._words_iter = iter(self.file.vocab)
        self._vectors_file.seek(0)

    def _read_word(self) -> str:
        return next(self._words_iter)

    def _read_vector(self) -> VectorType:
        vec_bytes = self._vectors_file.read(self._vector_size_in_bytes)
        vector = numpy.frombuffer(vec_bytes, dtype=self.dtype)
        return numpy.asarray(vector, dtype=self.out_dtype)

    def _skip_vector(self) -> None:
        self._vectors_file.seek(self._vector_size_in_bytes, io.SEEK_CUR)


class _VectorsFileWrapper:
    """ Wraps vectors.bin file and allows to read each vector by index """

    def __init__(self, binary_file, dtype: DType, out_dtype: DType, vector_size: int):
        self.file = binary_file
        self.dtype = numpy.dtype(dtype)
        self.out_dtype = numpy.dtype(out_dtype or dtype)
        self._vector_size_in_bytes = vector_size * self.dtype.itemsize

    def __getitem__(self, index) -> VectorType:
        """ Returns a vector by its index in the file (random access). """
        file = self.file
        file.seek(index * self._vector_size_in_bytes, io.SEEK_SET)
        vec = numpy.frombuffer(file.read(self._vector_size_in_bytes), dtype=self.dtype)
        return vec.astype(self.out_dtype)

    def close(self):
        self.file.close()


class VVMEmbFile(EmbFile, Word2Vector):
    """
    (Custom format) A tar file storing vocabulary, vectors and metadata in 3 separate files.

    Features:

    #. the vocabulary can be loaded very quickly (with no need for an external vocab file) and it is
       loaded in memory when the file is opened;

    #. direct access to vectors

       - by word using :meth:`__getitem__` (e.g. ``file['hello']``)
       - by index using :meth:`vector_at`

    #. implements :meth:`__contains__` (e.g. ``'hello' in file``)

    #. all the information needed to open the file are stored in the file itself

    **Specifics.** The files contained in a VVM file are:

    - *vocab.txt*: contains each word on a separate line
    - *vectors.bin*: contains the vectors in binary format (concatenated)
    - *meta.json*: must contain (at least) the following fields:

      - *vocab_size*: number of word vectors in the file
      - *vector_size*: length of a word vector
      - *encoding*: text encoding used for vocab.txt
      - *dtype*: vector data type string (notation used by numpy)

    Attributes:
        path
        encoding
        dtype
        out_dtype
        verbose
        vocab (OrderedDict[str, int])
            map each word to its index in the file

    """
    DEFAULT_EXTENSION = '.vvm'

    def __init__(self, path: PathType, out_dtype: Optional[DType] = None,
                 verbose: int = DEFAULT_VERBOSE):
        """
        Args:
            path:
            out_dtype:
            verbose:
        """
        super().__init__(path, out_dtype, verbose=verbose)

        if not tarfile.is_tarfile(path):
            raise ValueError('not a valid vvm file: %s' % path)

        self._tar = tar = tarfile.open(path)

        def _open_tar_member(filename) -> IO[bytes]:
            member = tar.extractfile(filename)
            if member is None:
                raise ValueError('missing file inside the archive: ' + filename)
            return member

        # Read metadata
        with _open_tar_member(META_FILENAME) as meta_file:
            metadata = json.load(meta_file)
        self.metadata = metadata
        self.vocab_size = metadata['vocab_size']  # type: int
        self.vector_size = metadata['vector_size']
        self.encoding = metadata['encoding']
        self.dtype = metadata['dtype'] = numpy.dtype(metadata['dtype'])

        if self.out_dtype is None:  # type: ignore
            self.out_dtype = self.dtype

        # Extract vectors.bin
        self._vectors_wrapper = self._get_vectors_file_wrapper()
        self._vector_size_in_bytes = self.dtype.itemsize * self.vector_size

        # Load the vocabulary
        with _open_tar_member(VOCAB_FILENAME) as vocab_file:
            vocab_reader = io.TextIOWrapper(vocab_file, encoding=self.encoding)
            lines_iterable = maybe_progbar(vocab_reader, yes=verbose,
                                           total=self.vocab_size,
                                           desc='Loading the vocabulary')
            self.vocab = OrderedDict((line[:-1], index)
                                     for index, line in enumerate(lines_iterable))

    def _get_vectors_file_wrapper(self) -> '_VectorsFileWrapper':
        vectors_file = self._tar.extractfile(VECTORS_FILENAME)
        return _VectorsFileWrapper(vectors_file, self.dtype, self.out_dtype, self.vector_size)

    @overrides
    def _close(self) -> None:
        self._vectors_wrapper.close()
        self._tar.close()

    @overrides
    def _reader(self) -> VVMEmbFileReader:
        vectors_file = self._tar.extractfile(VECTORS_FILENAME)
        return VVMEmbFileReader(self, vectors_file)

    @overrides
    def _loader(self, words: Iterable[str], missing_ok=True, verbose: Optional[bool] = None):
        word2index = self.vocab
        index2vec = self._get_vectors_file_wrapper()
        word2vec = MappingComposition(word2index, index2vec)
        return RandomAccessLoader(words, word2vec=word2vec,
                                  word2index=self.vocab.__getitem__,
                                  missing_ok=missing_ok,
                                  verbose=coalesce(verbose, self.verbose),
                                  close_hook=lambda: index2vec.close())

    @overrides
    def words(self) -> Iterable[str]:
        return self.vocab.keys()

    def __contains__(self, word: str) -> bool:
        """ Returns True if the file contains a vector for ``word`` """
        return word in self.vocab

    def vector_at(self, index: int) -> VectorType:
        """ Returns a vector by its index in the file (random access). """
        if index >= self.vocab_size or index < -self.vocab_size:
            raise IndexError(index)
        if index < 0:
            index += self.vocab_size
        return self._vectors_wrapper[index]

    def __getitem__(self, word) -> VectorType:
        """ Returns the vector associated to a word (random access to file). """
        index = self.vocab[word]
        return self._vectors_wrapper[index]

    @classmethod
    def _create(cls, out_path: Path, word_vectors: Iterable[Tuple[str, VectorType]],
                vector_size: int, vocab_size: Optional[int],
                compression: Optional[str] = None, verbose: bool = True,
                encoding: str = DEFAULT_ENCODING,
                dtype: Optional[DType] = None) -> Path:

        echo = print if verbose else noop
        if not dtype:
            (_, vector), word_vectors = glance_first_element(word_vectors)
            dtype = vector.dtype
        else:
            dtype = numpy.dtype(dtype)

        # Write everything in a temporary directory and then pack them into a tar file
        tempdir = Path(tempfile.mkdtemp())
        vocab_tmp_path = tempdir / VOCAB_FILENAME
        vectors_tmp_path = tempdir / VECTORS_FILENAME
        meta_tmp_path = tempdir / META_FILENAME

        with open(vocab_tmp_path, 'wt', encoding=encoding) as vocab_file, \
            open(vectors_tmp_path, 'wb') as vectors_file:  # noqa

            desc = 'Generating {} and {} file'.format(VOCAB_FILENAME, VECTORS_FILENAME)
            i = -1
            for i, (word, vector) in progbar(enumerate(word_vectors), verbose, desc=desc,
                                             total=vocab_size):
                if '\n' in word:
                    raise ValueError("the word number %d contains one or more newline characters: "
                                     "%r" % (i, word))
                vocab_file.write(word)
                vocab_file.write('\n')

                check_vector_size(i, vector, vector_size)
                vectors_file.write(numpy.asarray(vector, dtype).tobytes())

        actual_vocab_size = i + 1
        warn_if_wrong_vocab_size(vocab_size, actual_vocab_size,
                                 extra_info='the actual size will be written in meta.json')
        vocab_size = actual_vocab_size

        echo('Writing {}...'.format(META_FILENAME))
        metadata = {
            "vocab_size": vocab_size,
            "vector_size": vector_size,
            "dtype": dtype.str,
            "encoding": encoding
        }
        with open(meta_tmp_path, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=2)

        if not compression:
            tar_path = out_path
            tar_mode = 'w'
        elif compression in _TAR_COMPRESSIONS:
            tar_path = out_path
            tar_mode = 'w:' + compression
        else:
            warnings.warn('A VVM file is just a TAR file; you should compress it using '
                          'one the formats directly supported by tarfile ({}). '
                          'Using another compression format will require me to create a '
                          'temporary uncompressed TAR file first, doubling the required time!')
            tar_path = out_path.with_suffix(out_path.suffix + '.tmp')
            tar_mode = 'w'

        echo('Packing all the files together')
        with tarfile.open(tar_path, tar_mode) as tar_file:
            tar_file.add(str(vocab_tmp_path), VOCAB_FILENAME)
            tar_file.add(str(vectors_tmp_path), VECTORS_FILENAME)
            tar_file.add(str(meta_tmp_path), META_FILENAME)

        shutil.rmtree(tempdir)

        if compression and compression not in _TAR_COMPRESSIONS:
            echo("Compressing to %s file: %s" % (compression, out_path))
            with open_file(out_path, 'wb', compression=compression) as compressed_file:
                with open(tar_path, 'rb') as non_compressed_file:
                    shutil.copyfileobj(non_compressed_file, compressed_file)

            os.remove(tar_path)

        return out_path

    @classmethod
    def create(cls, out_path: PathType, word_vectors: PairsType, vocab_size: Optional[int] = None,
               compression: Optional[str] = None, verbose: bool = True, overwrite: bool = False,
               encoding: str = DEFAULT_ENCODING,
               dtype: Optional[DType] = None) -> None:
        """
        Format-specific arguments are encoding and dtype.

        Being VVM a tar file, you should use a compression supported by the tarfile package
        (avoid zip): gz, bz2 or xz.

        See :meth:`~embfile.core.file.EmbFile.create` for more doc.
        """
        super().create(out_path, word_vectors, vocab_size, compression, verbose, overwrite,
                       encoding=encoding, dtype=dtype)

    @classmethod
    def create_from_file(cls, source_file: 'EmbFile', out_dir: Optional[PathType] = None,
                         out_filename: Optional[str] = None, vocab_size: Optional[int] = None,
                         compression: Optional[str] = None, verbose: bool = True,
                         overwrite: bool = False, encoding: str = DEFAULT_ENCODING,
                         dtype: Optional[DType] = None) -> Path:
        """
        Format-specific arguments are encoding and dtype.
        Being VVM a tar file, you should use a compression supported by the tarfile package
        (avoid zip): gz, bz2 or xz.

        See :meth:`~embfile.core.file.EmbFile.create_from_file` for more doc.
        """
        return super().create_from_file(
            source_file, out_dir, out_filename, vocab_size, compression,
            verbose, overwrite, encoding=encoding, dtype=dtype)
