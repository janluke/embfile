"""
Tests for base classes EmbFileCreator and AbstractEmbFileReader + WordVector.
"""
import gzip
from itertools import zip_longest
from pathlib import Path

import numpy
import pytest

from embfile.errors import IllegalOperation
from tests.conftest import REFERENCE_VECTOR_SIZE, REFERENCE_VOCAB_SIZE
from tests.mocks import MockEmbFile, MockEmbFileReader


class TestBaseCreateMethod:

    def test_create_with_list(self, reference_pairs):
        MockEmbFile.create('fake_path', reference_pairs)
        _create_called_with = MockEmbFile._create_kwargs
        assert _create_called_with.vocab_size == REFERENCE_VOCAB_SIZE
        assert _create_called_with.vector_size == REFERENCE_VECTOR_SIZE
        assert _create_called_with.pairs == reference_pairs

    def test_create_with_dict(self, reference_dict):
        MockEmbFile.create('fake_path', reference_dict)
        _create_called_with = MockEmbFile._create_kwargs
        assert _create_called_with.vocab_size == REFERENCE_VOCAB_SIZE
        assert _create_called_with.vector_size == REFERENCE_VECTOR_SIZE
        assert _create_called_with.pairs == reference_dict.items()

    def test_create_with_iterator(self, reference_pairs):
        MockEmbFile.create('fake_path', iter(reference_pairs))
        _create_called_with = MockEmbFile._create_kwargs
        assert _create_called_with.vocab_size is None
        assert _create_called_with.vector_size == REFERENCE_VECTOR_SIZE
        assert list(_create_called_with.pairs) == reference_pairs

    def test_create_with_wrong_type(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            MockEmbFile.create('fake_path', word_vectors=5)

    def test_create_warns_for_wrong_vocab_size(self, reference_pairs):
        wrong_vocab_size = REFERENCE_VOCAB_SIZE + 5
        with pytest.warns(UserWarning):
            MockEmbFile.create('fake_path', reference_pairs, vocab_size=wrong_vocab_size)
        _create_called_with = MockEmbFile._create_kwargs
        assert _create_called_with.vocab_size == REFERENCE_VOCAB_SIZE

    def test_create_from_file_returned_path(self, tmpdir):
        src_file = MockEmbFile(path='path/to/file.txt')

        # default out_dir and out_filename
        out_path = MockEmbFile.create_from_file(src_file, compression='gz')
        assert out_path == Path('path/to/file.mock.gz')

        # custom out_dir and filename
        out_path = MockEmbFile.create_from_file(src_file, out_dir='out/dir/',
                                                out_filename='out_filename.mock')
        assert out_path == Path('out/dir/out_filename.mock')

    def test_create_from_file_returned_path_with_compressed_source_file(self, tmpdir):
        # We need to create a real compressed file because EmbFile decompresses it
        path = tmpdir / 'file.txt.gz'
        with gzip.open(path, 'wt') as f:
            f.write('ciao')
        compressed_src_file = MockEmbFile(path=path)

        # default out_dir and out_filename + source file is compressed
        out_path = MockEmbFile.create_from_file(compressed_src_file, compression='bz2')
        assert out_path == Path(tmpdir / 'file.mock.bz2')

        # non-default filename arg
        out_path = MockEmbFile.create_from_file(compressed_src_file, out_filename='changed',
                                                compression='bz2')
        assert out_path == Path(tmpdir / 'changed')

    def test_create_with_existing_file(self, tmp_path):
        path = tmp_path / 'existing'
        path.touch()
        data = [('blah', numpy.array([1, 2]))]
        with pytest.raises(FileExistsError):
            MockEmbFile.create(path, data, overwrite=False)
        MockEmbFile.create(path, data, overwrite=True)


class TestAbstractEmbFileReader:

    @pytest.fixture(scope='function')
    def reader(self, reference_pairs):
        return MockEmbFileReader(reference_pairs)

    def test_iteration(self, reader, reference_pairs):
        for word, (target_word, target_vector) in zip_longest(reader, reference_pairs):
            assert word == target_word
            assert numpy.allclose(reader.current_vector, target_vector)

    def test_current_vector_raises(self, reader):
        with pytest.raises(IllegalOperation):
            reader.current_vector

    def test_reset(self, reader, reference_pairs):
        reader.next_word()
        reader.next_word()
        reader.reset()
        first_word, first_vector = reference_pairs[0]
        with pytest.raises(IllegalOperation):
            reader.current_vector
        assert reader.next_word() == first_word
        assert numpy.allclose(reader.current_vector, first_vector)

    def test_close_post_conditions(self, reader):
        reader.close()
        with pytest.raises(IllegalOperation):
            reader.next_word()
        with pytest.raises(IllegalOperation):
            reader.current_vector
