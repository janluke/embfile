__all__ = ['BinaryEmbFile', 'BinaryEmbFileReader']

import io
import mmap
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Optional, Tuple

import numpy
from overrides import overrides

from embfile._utils import noop, progbar
from embfile.compression import open_file
from embfile.core import AbstractEmbFileReader, EmbFile
from embfile.core._file import (DEFAULT_VERBOSE, check_vector_size, glance_first_element,
                                warn_if_wrong_vocab_size)
from embfile.errors import BadEmbFile
from embfile.types import DType, PairsType, PathType, VectorType

#: Default text encoding
DEFAULT_ENCODING = 'utf-8'

#: Default vector data type (little-endian single-precision floating point numbers)
DEFAULT_DTYPE = numpy.dtype('<f4')


def _bom_free_version(encoding: str) -> str:
    """ Given an utf encoding, returns a BOM-free version of it (little-endian version) """

    def utf_aliases(num):
        return {fmt % num for fmt in ['u%d', 'utf%d', 'utf-%d', 'utf_%d']}

    encoding = encoding.lower()
    if encoding in utf_aliases(16):
        return 'utf_16_le'
    if encoding in utf_aliases(32):
        return 'utf_32_le'
    return encoding


def _take_until_delimiter(delim: str,
                          byte_array: bytes,
                          start_position: int,
                          encoding: str,
                          max_bytes: Optional[int] = None) -> Tuple[str, int]:
    """
    Reads the text that precedes a delimiter character and returns it along with the index of the
    first byte after the delimiter.
    Used by _read_until_delimiter, factored out for easier testing (this can be tested on bytes
    array, no need for creating a mmap object).

    Implementation Notes
    --------------------
    Since the encoding could be a multy-byte encoding (e.g. utf-16), we can't simply
    perform a search of the encoded delimiter at the binary level (searching bytes in bytes),
    we have to consider "character boundaries".
    Indeed, if the delimiter is encoded as 'x' and there's one character in the stream
    encoded as 'yx' before the actual delimiter, a simple find('x') would return a
    "false positive".

    Unfortunately, reading and (incrementally) decoding one byte at a time is incredible
    slow in Python. Decoding in chunk is better, but still really slow.

    For this reasons and because the bug described above is very rare (or impossible
    for 1-byte encodings), I first search the delimiter in the "byte space" ignoring
    character boundaries;
    then I check whether decoding the bytes that precede the delimiter produces a
    UnicodeDecodeError due to "truncated data". If it does, we know we found a false
    positives; in that case:

    1. we find the end of the truncated character which starts at ``decode_error.start``
    2. we continue our search of the delimiter bytes after the character end.
    """
    delim_bytes = delim.encode(encoding)

    find_start = start_position
    if max_bytes is not None:
        find_end = start_position + max_bytes

    while True:
        if byte_array[find_start:find_start + 1] == b'':
            return '', find_start

        if max_bytes is None:
            delim_start = byte_array.find(delim_bytes, find_start)
        else:
            delim_start = byte_array.find(delim_bytes, find_start, find_end)

        if delim_start < 0:
            if max_bytes:
                msg = ("expected delimiter %r wasn't found from position %d after %d bytes read "
                       "(max number of bytes allowed)." % (delim, start_position, max_bytes))
            else:
                msg = ("expected delimiter %r wasn't found from position %d to the end of the file."
                       % (delim, start_position))

            raise BadEmbFile(msg)

        delim_end = delim_start + len(delim_bytes)
        try:
            text = byte_array[start_position:delim_start].decode(encoding)
            return text, delim_end
        except UnicodeDecodeError as err:
            # adjust limits of the error to make it relative to byte_array start
            err.start += start_position
            err.end += start_position

            if err.end == delim_start:  # err.reason == "truncated data"
                # False positive: delimiter's bytes starts inside another character's bytes
                # Find the end of the truncated character
                char_start = err.start
                for char_end in range(char_start + 1, char_start + 4):
                    try:
                        byte_array[char_start:char_end].decode(encoding)
                    except UnicodeDecodeError as char_err:
                        char_err.start += char_start
                        char_err.end += char_start

                        if char_err.end != char_end:
                            char_err.object = byte_array
                            raise char_err
                    else:
                        # continue the search of the delimiter after the character
                        find_start = char_end
                        break
            else:
                # the error has nothing to do with our imperfect search of the delimiter
                err.object = byte_array
                raise err


def _read_until_delimiter(delim: str,
                          mem_map: mmap.mmap,
                          encoding: str,
                          max_bytes: Optional[int] = None) -> str:
    """
    Reads the text that precedes a delimiter character and returns it. If there are not bytes to
    read, returns the empty string ''.

    Raises:
        UnicodeDecodeError
        BadEmbFile:
            if the delimiter character is not found before the end of the file (or before
            ``max_bytes``, if provided), it raises ``BadEmbFile`` error.
    """
    text, new_position = _take_until_delimiter(delim, mem_map, mem_map.tell(),  # type: ignore
                                               encoding, max_bytes)
    mem_map.seek(new_position, io.SEEK_SET)
    return text


class BinaryEmbFileReader(AbstractEmbFileReader):
    """ :class:`~embfile.core.EmbFileReader` for the binary format. """

    #: Conservative upper bound for the length (in bytes) of the header of a binary embedding file
    _MAX_HEADER_BYTES = 128
    #: Conservative upper bound for the length (in bytes) of a word
    _MAX_WORD_BYTES = 1024

    def __init__(self, file_obj: BinaryIO,
                 encoding: str = DEFAULT_ENCODING,
                 dtype: DType = DEFAULT_DTYPE,
                 out_dtype: Optional[DType] = None):
        super().__init__(out_dtype or dtype)

        self.dtype = numpy.dtype(dtype)
        encoding = _bom_free_version(encoding)
        self.encoding = encoding

        self._file_obj = file_obj
        self._mmap = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)

        self.header = self._read_header()
        self._body_start = self._mmap.tell()  # store the position where the actual data starts

        self._vector_size_in_bytes = self.dtype.itemsize * self.header['vector_size']

    @classmethod
    def from_path(cls, path: PathType,
                  encoding: str = DEFAULT_ENCODING,
                  dtype: DType = DEFAULT_DTYPE,
                  out_dtype: Optional[DType] = None):
        return cls(open_file(path, 'rb'), encoding=encoding, dtype=dtype, out_dtype=out_dtype)

    def _read_header(self) -> Dict[str, Any]:
        header_line = _read_until_delimiter('\n', self._mmap, self.encoding, self._MAX_HEADER_BYTES)
        vocab_size, vector_size = map(int, header_line.split())
        return {'vocab_size': vocab_size, 'vector_size': vector_size}

    @overrides
    def _close(self) -> None:
        self._mmap.close()
        self._file_obj.close()

    @overrides
    def _reset(self) -> None:
        self._mmap.seek(self._body_start, io.SEEK_SET)

    @overrides
    def _read_word(self) -> str:
        word = _read_until_delimiter(' ', self._mmap, self.encoding, self._MAX_WORD_BYTES)
        if not word:
            raise StopIteration
        return word

    @overrides
    def _read_vector(self) -> VectorType:
        vec_bytes = self._mmap.read(self._vector_size_in_bytes)
        vector = numpy.frombuffer(vec_bytes, dtype=self.dtype)
        return numpy.asarray(vector, dtype=self.out_dtype)

    @overrides
    def _skip_vector(self) -> None:
        self._mmap.seek(self._vector_size_in_bytes, io.SEEK_CUR)


class BinaryEmbFile(EmbFile):
    """
    Format used by the Google word2vec tool.
    You can use it to read the file `GoogleNews-vectors-negative300.bin
    <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing>`_.

    It begins with a text header line of space-separated fields::

        <vocab_size> <vector_size>\n

    Each word vector pair is encoded as following:

    - encoded word + space
    - followed by the binary representation of the vector.

    Attributes:
        path
        encoding
        dtype
        out_dtype
        verbose

    """
    DEFAULT_EXTENSION = '.bin'

    def __init__(self, path: PathType, encoding: str = DEFAULT_ENCODING,
                 dtype: DType = DEFAULT_DTYPE, out_dtype: Optional[DType] = None,
                 verbose: int = DEFAULT_VERBOSE):
        """
        Args:
            path:
                path to the (eventually compressed) file
            encoding:
                text encoding; **note:** if you provide an utf encoding (e.g. *utf-16*) that uses a
                BOM (Byte Order Mark) without specifying the byte-endianness (e.g. *utf-16-le* or
                *utf-16-be*), the little-endian version is used (*utf-16-le*).
            dtype:
                a valid numpy data type (or whatever you can pass to numpy.dtype())
                (default: '<f4'; little-endian float, 4 bytes)
            out_dtype:
                all the vectors returned will be (eventually) converted to this data type;
                by default, it is equal to the original data type of the vectors in the file,
                i.e. no conversion takes place.

        """
        super().__init__(path, out_dtype, verbose=verbose)

        self.encoding = _bom_free_version(encoding)
        self.dtype = numpy.dtype(dtype)
        self.out_dtype = numpy.dtype(out_dtype) if out_dtype else self.dtype

        # Read the header
        with self.reader() as reader:
            self.vocab_size = reader.header['vocab_size']
            self.vector_size = reader.header['vector_size']

    @overrides
    def _reader(self) -> BinaryEmbFileReader:
        return BinaryEmbFileReader.from_path(
            path=self.path,
            encoding=self.encoding,
            dtype=self.dtype,
            out_dtype=self.out_dtype
        )

    @overrides
    def _close(self):
        pass

    @classmethod
    def _create(cls, out_path: Path,
                word_vectors: Iterable[Tuple[str, VectorType]],
                vector_size: int,
                vocab_size: Optional[int],
                compression: Optional[str] = None,
                verbose: bool = True,
                encoding: str = DEFAULT_ENCODING,
                dtype: Optional[DType] = None) -> Path:

        echo = print if verbose else noop
        encoding = _bom_free_version(encoding)
        if not dtype:
            (_, first_vector), word_vectors = glance_first_element(word_vectors)
            dtype = first_vector.dtype
        else:
            dtype = numpy.dtype(dtype)

        if not vocab_size:
            raise ValueError('unable to infer vocab_size; you must manually provide it')

        with open_file(out_path, 'wb', compression=compression) as file:
            header_line = '%d %d\n' % (vocab_size, vector_size)
            echo('Writing the header: %s', header_line)
            header_bytes = header_line.encode(encoding)
            file.write(header_bytes)

            for i, (word, vector) in progbar(enumerate(word_vectors), verbose, total=vocab_size):
                if ' ' in word:
                    raise ValueError("the word number %d contains one or more spaces: %r"
                                     % (i, word))
                file.write((word + ' ').encode(encoding))

                check_vector_size(i, vector, vector_size)
                file.write(numpy.asarray(vector, dtype).tobytes())

        warn_if_wrong_vocab_size(vocab_size, actual_size=i + 1,
                                 extra_info='As a consequence, the header of the file has a wrong '
                                            'vocab_size')
        return out_path

    @classmethod
    def create(cls, out_path: PathType, word_vectors: PairsType, vocab_size: Optional[int] = None,
               compression: Optional[str] = None, verbose: bool = True, overwrite: bool = False,
               encoding: str = DEFAULT_ENCODING,
               dtype: Optional[DType] = None) -> None:
        """
        Format-specific arguments are ``encoding`` and ``dtype``.

        **Note:** all the text is encoded without BOM (Byte Order Mark). If you pass
        "utf-16" or "utf-18", the little-endian version is used (e.g. "utf-16-le")

        See :meth:`~embfile.core.file.EmbFile.create` for more.
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
        Format-specific arguments are ``encoding`` and ``dtype``.

        **Note:** all the text is encoded without BOM (Byte Order Mark). If you pass
        "utf-16" or "utf-18", the little-endian version is used (e.g. "utf-16-le")

        See :meth:`~embfile.core.file.EmbFile.create_from_file` for more.
        """
        return super().create_from_file(
            source_file, out_dir, out_filename, vocab_size, compression,
            verbose, overwrite, encoding=encoding, dtype=dtype)
