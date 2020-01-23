from pathlib import Path
from typing import Iterable, Optional, Tuple

import embfile
from embfile.types import DType, PathType, VectorType
from embfile.core import EmbFile, EmbFileReader



# TODO: implement a reader
# Note: you could also extend AbstractEmbFileReader if it's convenient for you
class CustomEmbFileReader(EmbFileReader):
    """ Implements file sequential reading """
    def __init__(self, out_dtype: DType):  # TODO: add the needed arguments
        super().__init__(out_dtype)

    def _close(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def next_word(self) -> str:
        pass

    def current_vector(self) -> VectorType:
        pass


@embfile.register_format('custom', extensions=['.cst', '.cust'])
class CustomEmbFile(EmbFile):

    def __init__(self, path: PathType, out_dtype: DType = None, verbose: int = 1):
        super().__init__(path, out_dtype, verbose)  # this is not optional
        # cls.vocab_size = ??
        # cls.vector_size = ??

    def _close(self) -> None:
        pass

    def _reader(self) -> EmbFileReader:
        return CustomEmbFileReader()  # TODO: pass the needed arguments

    # Optional:
    def _loader(self, words: Iterable[str], missing_ok: bool = True, verbose: Optional[int] = None) -> 'VectorsLoader':
        """ By default, a SequentialLoader is returned. """
        return super()._loader(words, missing_ok, verbose)

    @classmethod
    def _create(cls, out_path: Path, word_vectors: Iterable[Tuple[str, VectorType]],
                vector_size: int, vocab_size: Optional[int], compression: Optional[str] = None,
                verbose: bool = True, **format_kwargs) -> None:
        pass


if __name__ == '__main__':
    print(embfile.FORMATS)
