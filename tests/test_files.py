from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy
import pytest

import embfile
from embfile import BinaryEmbFile, TextEmbFile, VVMEmbFile
from embfile._utils import sample_dict_subset
from embfile.compression import default_compression_ext
from embfile.core import EmbFile, WordVector
from embfile.errors import IllegalOperation
from embfile.formats.bin import _take_until_delimiter
from tests.conftest import REFERENCE_VOCAB_SIZE, generate_find_case
from tests.mocks import MockEmbFile


def assert_equiv_word_vectors(actual, target):
    assert actual.word == target.word
    assert numpy.allclose(actual.vector, target.vector)


def assert_equiv_word_vector_lists(actual_pairs, target_pairs):
    assert len(actual_pairs) == len(target_pairs)
    for actual, target in zip(actual_pairs, target_pairs):
        assert_equiv_word_vectors(actual, target)


def assert_equiv_word2vec(actual, target):
    assert actual.keys() == target.keys()
    for word, target_vector in target.items():
        assert numpy.allclose(actual[word], target_vector)


# noinspection PyMethodMayBeStatic
class BaseTestEmbFile:
    """
    Tests file creation and loading jointly:
    - first generate some target (word, vector) pairs
    - create an embedding file containing those pairs
    - check that file.to_list() is equivalent to the target pairs.

    This will work only if both file creation and file reading works correctly.

    The fixture ``pairs_and_file`` creates some word_vectors and a file containing them (hopefully).
    You can specify for what encodings, dtypes and compression formats to test using the
    fixtures ``encoding``, ``dtype`` and ``compressions`` and passing them as dependencies
    to ``pairs_and_file``. The test depending on ``pairs_and_file`` will be repeated
    for each possible combination of those parameters. Testing "exhaustively"
    may seem useless and wasteful but I actually discovered a bug in the Python library
    doing that.

    Testing methods that don't require to be called for multiple combinations of encoding,
    dtypes or whatever, can use the fixture ``file`` which is a file containing the
    ``reference_pairs`` used through the entire test session (see conftest.py).
    """
    CLS: Type[EmbFile]  # the EmbFile subclass under test
    CREATE_KWARGS: Dict[str, Any] = dict()  # kwargs to pass to create() in the fixture "file"

    @pytest.fixture(params=['ascii', 'utf-8', 'utf-16'])
    def encoding(self, request):
        """ Encodings to test for in test_creation_and_loading_jointly """
        return request.param

    @pytest.fixture(params=['<f4', '>f8', '<i4'])
    def dtype(self, request):
        """ DTypes to test for in test_creation_and_loading_jointly """
        return request.param

    @pytest.fixture(params=[None, 'gz', 'xz', 'zip'])
    def compression(self, request):
        """ Compressions to test for in test_creation_and_loading_jointly """
        return request.param

    @pytest.fixture
    def pairs_and_file(self, tmp_path,
                       pairs_factory) -> Tuple[List[WordVector], EmbFile]:
        """
        Generates some target word_vectors, creates a file containing them and opens the file.
        Returns the word_vectors and the file. Test sub-classes must implement this. In subclass,
        include fixtures like encoding, dtype and compression in the argument list of this
        fixture to test
        for every combination of those.

        Used by test_creation_and_loading_jointly.
        """
        raise NotImplementedError

    def test_creation_and_loading_jointly(self, pairs_and_file):
        target_pairs, file = pairs_and_file
        assert file.vocab_size == len(target_pairs)
        assert file.vector_size == len(target_pairs[0].vector)
        assert_equiv_word_vector_lists(file.to_list(), target_pairs)

    @pytest.fixture
    def file(self, tmp_path, reference_pairs) -> EmbFile:
        """ Create a file for the reference_pairs and returns it. This fixture can be used by tests
        that don't need to be performed for multiple combinations of encoding and dtype """
        path = tmp_path / ('reference_file.' + self.CLS.DEFAULT_EXTENSION)
        self.CLS.create(path, reference_pairs, overwrite=True, **self.CREATE_KWARGS)
        return self.CLS(path)

    def test_to_dict(self, file, reference_dict):
        actual_dict = file.to_dict()
        assert_equiv_word2vec(actual_dict, reference_dict)

    def test_words(self, file, reference_words):
        assert tuple(file.words()) == reference_words

    def test_vectors(self, file, reference_vectors):
        vectors = list(file.vectors())
        assert len(vectors) == len(reference_vectors)
        for vector, target in zip(vectors, reference_vectors):
            assert numpy.allclose(vector, target)

    def test_load(self, file, reference_dict):
        target_word2vec = sample_dict_subset(reference_dict, 10)
        words_to_load = list(target_word2vec.keys())
        # All present words
        word2vec = file.load(words_to_load)
        assert_equiv_word2vec(word2vec, target_word2vec)
        # One word missing
        with pytest.raises(KeyError):
            file.load(words_to_load + ['<<missing-word>>'])

    @pytest.mark.parametrize('num_missing', [0, 100])
    @pytest.mark.parametrize('num_present', [0, REFERENCE_VOCAB_SIZE // 2, REFERENCE_VOCAB_SIZE])
    def test_find(self, file, reference_dict, num_present, num_missing):
        case = generate_find_case(reference_dict, num_present, num_missing)
        word2vec, actual_missing = file.find(case.query_words)
        assert actual_missing == case.missing_words
        assert_equiv_word2vec(word2vec, case.target_dict)

    @pytest.mark.parametrize('num_present', [0, REFERENCE_VOCAB_SIZE // 2])
    def test_filter(self, file, reference_dict, num_present):
        target = sample_dict_subset(reference_dict, num_present)

        def condition(word):
            return word in target

        actual = dict(file.filter(condition))
        assert_equiv_word2vec(actual, target)

    def test_close(self, file):
        reader1 = file.reader()
        reader2 = file.reader()
        file.close()
        with pytest.raises(IllegalOperation):
            next(reader1)
        with pytest.raises(IllegalOperation):
            next(reader2)
        with pytest.raises(IllegalOperation):
            file.reader()

    def test_reader_reset(self, file):
        with file.reader() as reader:
            words = list(reader)
            reader.reset()
            words2 = list(reader)
            assert words == words2

    @pytest.mark.parametrize('encoding', ['utf-8', 'utf-16'])
    def test_save_vocab(self, file, encoding):
        def compare_vocabs(file, vocab_path):
            with open(vocab_path, 'rt', encoding=encoding) as vocab_file:
                vocab_words = [line.rstrip() for line in vocab_file]
            for file_word, vocab_word in zip_longest(file.words(), vocab_words):
                assert file_word == vocab_word

        vocab_path = file.save_vocab(encoding=encoding)
        compare_vocabs(file, vocab_path)

        with pytest.raises(FileExistsError):
            file.save_vocab()

        file.save_vocab(overwrite=True)

    def test_create_raises_for_inconsistent_vector_size(self, tmp_path):
        pairs = {'hello': numpy.array([1, 2]), 'world': [3, 4], 'how': [1, 2, 3]}
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            self.CLS.create(tmp_path / 'file', pairs, overwrite=True)

    def test_create_warns_for_wrong_vocab_size_when_vocab_is_iterator(self, tmp_path):
        pairs = {'hello': numpy.array([1, 2]), 'world': [3, 4], 'how': [1, 2]}

        with pytest.warns(UserWarning):
            self.CLS.create(tmp_path / 'file', word_vectors=iter(pairs.items()),
                            vocab_size=len(pairs) + 2, overwrite=True)

        with pytest.warns(UserWarning):
            self.CLS.create(tmp_path / 'file', word_vectors=iter(pairs.items()),
                            vocab_size=len(pairs) - 1, overwrite=True)

    def test_repr(self, file):
        repr(file)  # just ensure it runs with no error


class TestEmbFileUsingMock(BaseTestEmbFile):
    """ This will test the base class EmbFile """
    CLS = MockEmbFile

    @pytest.fixture
    def pairs_and_file(self, tmp_path, pairs_factory, dtype):
        target_pairs = pairs_factory()
        path = tmp_path / 'file.mock'
        with MockEmbFile(target_pairs, path=path) as file:
            yield target_pairs, file

    @pytest.fixture
    def file(self, tmp_path, reference_pairs):
        return MockEmbFile(reference_pairs, path=tmp_path / 'file.mock')

    def test_create_raises_for_inconsistent_vector_size(self, tmp_path):
        pass

    def test_create_warns_for_wrong_vocab_size_when_vocab_is_iterator(self, tmp_path):
        pass


class TestTextEmbFile(BaseTestEmbFile):
    CLS = TextEmbFile
    CREATE_KWARGS = dict(precision=10)

    @pytest.fixture
    def pairs_and_file(self, tmp_path, pairs_factory, encoding, dtype, compression):
        target_pairs = pairs_factory(encoding=encoding, dtype=dtype)
        path = tmp_path / 'embfile.txt'
        if compression:
            path = path.with_suffix(path.suffix + '.' + compression)

        TextEmbFile.create(path, target_pairs, encoding=encoding, precision=10,
                           compression=compression, overwrite=True)

        with TextEmbFile(path, encoding=encoding, out_dtype=dtype) as file:
            yield target_pairs, file

    def test_open_with_no_header(self, tmp_path):
        path = tmp_path / 'file.txt'
        path.write_text('ciao 1 2 3 4 5\n'
                        'mamma 6 7 8 9 10')
        with TextEmbFile(path) as file:
            assert file.vector_size == 5
            assert file.vocab_size is None
            target_pairs = [
                WordVector('ciao', numpy.array([1, 2, 3, 4, 5])),
                WordVector('mamma', numpy.array([6, 7, 8, 9, 10]))
            ]
            assert_equiv_word_vector_lists(file.to_list(), target_pairs)


class TestBinaryEmbFile(BaseTestEmbFile):
    CLS = BinaryEmbFile

    @pytest.fixture
    def pairs_and_file(self, tmp_path, pairs_factory, encoding, dtype, compression) -> EmbFile:
        target_pairs = pairs_factory(encoding=encoding, dtype=dtype)
        path = tmp_path / 'embfile.bin'
        if compression:
            path = Path(str(path) + default_compression_ext(compression))
        BinaryEmbFile.create(path, target_pairs, encoding=encoding, dtype=dtype,
                             compression=compression, overwrite=True)
        uncompressed_path = (embfile.extract(path, dest_dir=path.parent, overwrite=True)
                             if compression
                             else path)
        with BinaryEmbFile(uncompressed_path, encoding=encoding, dtype=dtype) as file:
            yield target_pairs, file

    def test_take_until_delimiter_tricky_case(self):
        """ See _take_until_delimiter() docstring """
        delim = ' '
        delim_bytes = delim.encode('utf-16-le')
        # tricky case: the concatenation of char_1 and char_2 contains the delimiter's bytes:
        char_1 = (b'\x11' + delim_bytes[0:1]).decode('utf-16-le')
        char_2 = (delim_bytes[1:2] + b'\x11').decode('utf-16-le')

        target_text = '01234' + char_1 + char_2 + '789'
        input_text = target_text + delim + 'abcdef'
        input_bytes = input_text.encode('utf-16-le')
        target_end = len((target_text + delim).encode('utf-16-le'))

        for start_char in [0, 2]:
            skipped_text = target_text[:start_char]
            start_byte = len(skipped_text.encode('utf-16-le'))
            text, delim_end = _take_until_delimiter(delim, input_bytes, start_byte, 'utf-16-le')
            assert text == target_text[start_char:]
            assert delim_end == target_end


class TestVVMEmbFile(BaseTestEmbFile):
    CLS = VVMEmbFile

    @pytest.fixture
    def pairs_and_file(self, tmp_path, pairs_factory, encoding, dtype, compression) -> EmbFile:
        target_pairs = pairs_factory(encoding=encoding, dtype=dtype)
        path = tmp_path / 'embfile.vvm'
        if compression:
            path = Path(str(path) + default_compression_ext(compression))
        VVMEmbFile.create(path, target_pairs, encoding=encoding, dtype=dtype,
                          compression=compression, overwrite=True)
        uncompressed_path = (embfile.extract(path, dest_dir=path.parent, overwrite=True)
                             if compression
                             else path)
        with VVMEmbFile(uncompressed_path, out_dtype=dtype) as file:
            yield target_pairs, file

    def test_contains(self, file, reference_pairs):
        for word, _ in reference_pairs:
            assert word in file
        assert '<<missing>>' not in file

    def test_getitem(self, file: VVMEmbFile, reference_pairs):
        for word, vector in reference_pairs:
            assert numpy.allclose(file[word], vector)
        with pytest.raises(KeyError):
            file['<<missing>>']

    def test_vector_at(self, file: VVMEmbFile, reference_vectors):
        n = len(reference_vectors)
        for i in range(-n, n):
            assert numpy.allclose(file.vector_at(i), reference_vectors[i])
        for i in [-n - 1, n]:
            with pytest.raises(IndexError):
                file.vector_at(i)
