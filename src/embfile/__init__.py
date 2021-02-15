from embfile._embfile import (FORMATS, BuildMatrixOutput, associate_extension, build_matrix, open,
                              register_format)
from embfile.compression import extract_file as extract
from embfile.compression import extract_if_missing
from embfile.core import EmbFile
from embfile.formats import BinaryEmbFile, TextEmbFile, VVMEmbFile

__version__ = '0.1.1'

__all__ = [
    'open', 'build_matrix', 'BuildMatrixOutput',
    'EmbFile', 'BinaryEmbFile', 'TextEmbFile', 'VVMEmbFile',
    'FORMATS', 'register_format', 'associate_extension',
    'extract', 'extract_if_missing',
]
