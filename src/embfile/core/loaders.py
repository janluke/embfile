__all__ = ['VectorsLoader', 'SequentialLoader', 'RandomAccessLoader', 'Word2Vector']

import abc
from itertools import islice
from typing import Callable, Iterable, Iterator, Optional

from overrides import overrides

from embfile._utils import noop, progbar
from embfile.types import VectorType
from embfile.word_vector import WordVector

if False:  # for mypy
    from embfile.core._file import EmbFile


class VectorsLoader(abc.ABC, Iterator['WordVector']):
    def __init__(self, words: Iterable[str], missing_ok: bool = True):
        """
        *(Abstract class)* Iterator that, given some input words, looks for the corresponding
        vectors into the file and yields a word vector pair for each vector found; once the
        iteration stops, the attribute ``missing_words`` contains the set of words not found.

        Subclasses can load the word vectors in any order.

        Args:
            words:
                the words to load
            missing_ok:
                If ``False``, raises a KeyError if any input word is not in the file
        """
        self.words = words
        self.missing_ok = missing_ok

    @property
    @abc.abstractmethod
    def missing_words(self):
        """ The words that have still to be found; once the iteration stops, it's the set of
        the words that are in the input ``words`` but not in the file. """

    def __iter__(self) -> Iterator['WordVector']:
        return self

    def close(self):
        """ Closes eventual open resources (e.g. a reader). """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SequentialLoader(VectorsLoader):
    """
    A Loader that just scans the file from beginning to the end and yields a word vector pair
    when it meets a requested word. Used by txt and bin files. It's unable to tell if a word is
    in the file or not before having read the entire file.

    The progress bar shows the percentage of file that has been examined, not the number
    of yielded word vectors, so the iteration may stop before the bar reaches its 100%
    (in the case that all the input words are in the file).
    """

    def __init__(self, file: 'EmbFile', words: Iterable[str],
                 missing_ok: bool = True, verbose: bool = False):
        super().__init__(words, missing_ok)
        self.file = file
        self.verbose = verbose
        self._missing_words = set(words)
        self._reader = file.reader()
        self._progbar = progbar(enable=verbose, total=self.file.vocab_size,
                                desc='Reading')

    @property
    def missing_words(self):
        return self._missing_words

    def __next__(self) -> 'WordVector':
        missing_words = self._missing_words
        if not missing_words:
            self.close()
            raise StopIteration

        for word in self._reader:
            self._progbar.update()
            if word in missing_words:
                missing_words.remove(word)
                return WordVector(word, self._reader.current_vector)
        else:
            self.close()
            if self.missing_ok:
                raise StopIteration
            else:
                num_missing = len(missing_words)
                missing_list = (', '.join(islice(missing_words, 20))
                                + ('...' if num_missing > 20 else ''))
                raise KeyError('%d words are missing: %s' % (num_missing, missing_list))

    @overrides
    def close(self):
        self._progbar.close()
        self._reader.close()


class Word2Vector(abc.ABC):
    """ Maps a word to a vector """

    @abc.abstractmethod
    def __getitem__(self, word: str) -> VectorType:  # type: ignore
        pass

    @abc.abstractmethod
    def __contains__(self, word: str) -> bool:
        pass


class RandomAccessLoader(VectorsLoader):
    """
    A loader for files that can randomly access word vectors.
    If word2index is provided, the words are sorted by their position
    and the corresponded vectors are loaded in this order;
    I observed that this significantly improves the performance (with VVMEmbFile)
    (presumably due to buffering).
    """

    def __init__(self, words: Iterable[str],
                 word2vec: Word2Vector,
                 word2index: Optional[Callable[[str], int]] = None,
                 missing_ok: bool = True,
                 verbose: bool = False,
                 close_hook: Optional[Callable] = None):
        """
        Args:
            words:
            word2vec:
                object that implements ``word2vec[word]`` and ``word in word2vec``
            word2index:
                function that returns the index (position) of a word inside the file;
                this enables an optimization for formats like VVM that store vectors
                sequentially in the same file.
            missing_ok:
            verbose:
            close_hook:
                function to call when closing this loader
        """
        super().__init__(words, missing_ok)
        self.verbose = verbose
        self.word2vec = word2vec
        self.word2pos = word2index

        echo = print if verbose else noop
        missing_words = set(words)
        num_words = len(missing_words)
        available_words = []
        for word in progbar(self.words, enable=verbose, desc='Collecting all available words'):
            if word in word2vec:
                available_words.append(word)
                missing_words.remove(word)
            elif not missing_ok:
                raise KeyError('word not found in the file: ' + word)

        if word2index:
            echo('Sorting words based on their position in the file')
            available_words.sort(key=self.word2pos)

        self._missing_words = missing_words
        self._available_words = available_words
        self._iterator = (WordVector(word, word2vec[word]) for word in available_words)

        desc = 'Loading available vectors (%d of %d)' % (len(available_words), num_words)
        self._progbar = progbar(enable=verbose, total=len(available_words), desc=desc)
        self.close_hook = close_hook

    @property
    def missing_words(self):
        return self._missing_words

    def __next__(self):
        self._progbar.update()
        return next(self._iterator)

    def close(self):
        self._progbar.close()
        self.close_hook()
