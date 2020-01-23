import numpy
import pytest
from pytest import fixture

import embfile
from embfile import BuildMatrixOutput
from embfile.initializers import NormalInitializer, normal
from tests.conftest import REFERENCE_VOCAB_SIZE, generate_find_case
from tests.mocks import MockEmbFile, MockInitializer


@fixture(scope='module')
def file(reference_pairs):
    return MockEmbFile(reference_pairs)


def generate_build_matrix_case(word2vec, num_present, num_missing, start_index=0):
    find_case = generate_find_case(word2vec, num_present, num_missing)
    word2index = {word: i for i, word in enumerate(find_case.query_words, start_index)}
    return BuildMatrixOutput(None, word2index, find_case.missing_words)


def check_build_matrix_output(actual: BuildMatrixOutput,
                              target: BuildMatrixOutput,
                              word2vec, initializer):
    """ If initializer is a MockInitializer, checks it was properly called. """
    assert actual.word2index == target.word2index
    assert actual.missing_words == target.missing_words

    found_words = actual.found_words()
    for word in found_words:
        assert numpy.allclose(actual.vector(word), word2vec[word])

    if target.missing_words:
        if isinstance(initializer, MockInitializer):
            if found_words:
                assert initializer.fit_called
            for word in target.missing_words:
                assert numpy.all(actual.vector(word) == initializer.fill_value)


@pytest.mark.parametrize('initializer', [MockInitializer(), NormalInitializer(), normal()])
@pytest.mark.parametrize('num_missing', [0, 25])
@pytest.mark.parametrize('num_present', [REFERENCE_VOCAB_SIZE // 2, REFERENCE_VOCAB_SIZE])
def test_build_matrix_from_dict(file, reference_dict, initializer, num_present, num_missing):
    case = generate_build_matrix_case(reference_dict, num_present, num_missing)
    output = embfile.build_matrix(file, case.word2index, oov_initializer=initializer)
    check_build_matrix_output(output, case, reference_dict, initializer)


@pytest.mark.parametrize('start_index', [0, 3])
@pytest.mark.parametrize('num_missing', [0, 25])
@pytest.mark.parametrize('num_present', [REFERENCE_VOCAB_SIZE // 2, REFERENCE_VOCAB_SIZE])
def test_build_matrix_from_list(file, reference_dict, num_missing, num_present, start_index):
    case = generate_build_matrix_case(reference_dict, num_present, num_missing, start_index)
    vocab = list(case.word2index)
    initializer = MockInitializer()
    output = embfile.build_matrix(file, vocab, start_index=start_index,
                                  oov_initializer=initializer)
    check_build_matrix_output(output, case, reference_dict, initializer)


def test_build_matrix_raises_for_dict_with_empty_slots():
    word2vec = {
        'a': numpy.random.rand(5),
        'b': numpy.random.rand(5),
        'c': numpy.random.rand(5),
        'd': numpy.random.rand(5)
    }
    oov_vector = numpy.array([1, 2, 3, 4, 5])
    word2index = {'a': 1, 'c': 3, 'missing': 5}
    file = MockEmbFile(list(word2vec.items()))
    out = embfile.build_matrix(file, word2index,
                               oov_initializer=lambda shape: oov_vector)
    assert out.missing_words == {'missing'}
    out.word2index == word2index
    for i in [0, 2, 4]:
        assert numpy.allclose(out.matrix[i], numpy.zeros(5))
    assert numpy.allclose(out.matrix[1], word2vec['a'])
    assert numpy.allclose(out.matrix[3], word2vec['c'])
    assert numpy.allclose(out.matrix[5], oov_vector)


def test_build_matrix_raises_for_no_words(file):
    with pytest.raises(ValueError):
        embfile.build_matrix(file, dict())
    with pytest.raises(ValueError):
        embfile.build_matrix(file, list())


def test_build_matrix_raises_for_non_injective_dict(file, reference_words):
    words = reference_words[:5] + ('missing_1', 'missing_2', 'missing_3')
    word2row = {word: i for i, word in enumerate(words)}  # dict okay

    # duplication in available words
    invalid_dict = dict(word2row)
    invalid_dict[words[3]] = 1
    with pytest.raises(ValueError):
        embfile.build_matrix(file, invalid_dict)

    # duplication in missing words
    other_invalid = dict(word2row)
    other_invalid[words[7]] = 5
    with pytest.raises(ValueError):
        embfile.build_matrix(file, other_invalid)
