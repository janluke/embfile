import random
from types import SimpleNamespace

import numpy
import pytest

from embfile._utils import sample_dict_subset, shuffled
from embfile.core import WordVector

DEFAULT_DTYPE = numpy.dtype('<f4')
DEFAULT_ENCODING = 'utf-8'

REFERENCE_VECTOR_SIZE = 10
REFERENCE_VOCAB_SIZE = 100
NON_REFERENCE_WORDS = 100

UTF_CHARS = [chr(code) for code in (list(range(0x21, 0x7F)) + list(range(0xA1, 513)))]
CHARSETS = {
    'ascii': [chr(code) for code in range(33, 127)],
    'utf-8': UTF_CHARS,
    'utf-16': UTF_CHARS,
    'utf-32': UTF_CHARS,
    'latin_1': [chr(code) for code in (list(range(0x21, 0x7F)) + list(range(0xA1, 0x100)))],
}


def sample_word(alphabet, min_len=1, max_len=20):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(alphabet, k=length))


def generate_vocab(vocab_size, charset=UTF_CHARS, forbidden_words=set()):
    words = set()
    while len(words) < vocab_size:
        word = sample_word(charset)
        if (word not in words) and (word not in forbidden_words):
            words.add(word)
    return words


def generate_pairs(vocab_size=REFERENCE_VOCAB_SIZE, vector_size=REFERENCE_VECTOR_SIZE,
                   encoding=DEFAULT_ENCODING, dtype=DEFAULT_DTYPE):
    charset = CHARSETS[encoding]
    words = generate_vocab(vocab_size, charset)
    return [WordVector(word, (numpy.random.rand(vector_size) * 10 - 5).astype(dtype))
            for i, word in enumerate(words)]


@pytest.fixture
def pairs_factory():
    return generate_pairs


@pytest.fixture(scope='session')
def reference_pairs():
    """ Used for testing methods that don't need to be tested for multiple encodings or dtypes """
    return generate_pairs()


@pytest.fixture(scope='session')
def reference_dict(reference_pairs):
    return dict(reference_pairs)


@pytest.fixture(scope='session')
def reference_words(reference_dict):
    return tuple(reference_dict.keys())


@pytest.fixture(scope='session')
def reference_vectors(reference_dict):
    return tuple(reference_dict.values())


def generate_find_case(word2vec: dict, num_present_words: int, num_missing_words: int):
    """ Generate a test case for EmbFile.find() """
    target_dict = sample_dict_subset(word2vec, num_present_words)
    present_words = list(target_dict.keys())
    missing_words = generate_vocab(num_missing_words, forbidden_words=word2vec)
    query_words = shuffled(present_words + list(missing_words))
    return SimpleNamespace(
        query_words=query_words,        # words to find
        target_dict=target_dict,        # expected output word2vec dictionary
        present_words=present_words,    # subset of query_words in the file
        missing_words=missing_words)    # subset of query_words NOT in the file
