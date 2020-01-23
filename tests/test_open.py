from typing import Optional
from unittest.mock import patch

import pytest

import embfile
from embfile.compression import EXTENSION_TO_COMPRESSION
from embfile.registry import FormatsRegistry
from embfile.types import DType, PathType


class FakeFile:
    def __init__(self, path: PathType, out_dtype: Optional[DType] = None,
                 verbose: int = 1, **kwargs):
        self.path = path
        self.out_dtype = out_dtype
        self.verbose = verbose
        self.kwargs = kwargs


TextFile = type('TextFile', (FakeFile,), {})
BinaryFile = type('BinaryFile', (FakeFile,), {})
VVMFile = type('VVMFile', (FakeFile,), {})


def registry():
    reg = FormatsRegistry()
    reg.register_format(TextFile, 'txt', ['.txt', '.vec'])
    reg.register_format(BinaryFile, 'bin', ['.bin'])
    reg.register_format(VVMFile, 'vvm', ['.vvm'])
    return reg


@pytest.mark.parametrize('compression', [''] + list(EXTENSION_TO_COMPRESSION.keys()))
@patch('embfile._embfile.FORMATS', registry())
def test_open_with_valid_extension(compression):
    assert embfile._embfile.open('path/to/file.txt' + compression).__class__ == TextFile
    assert embfile._embfile.open('path/to/file.vec' + compression).__class__ == TextFile
    assert embfile._embfile.open('path/to/file.bin' + compression).__class__ == BinaryFile
    assert embfile._embfile.open('path/to/file.vvm' + compression).__class__ == VVMFile


@pytest.mark.parametrize('compression', ['', '.gz', '.zip'])
@patch('embfile._embfile.FORMATS', registry())
def test_open_with_unknown_extension(compression):
    path = 'path/to/file.unknown' + compression
    with pytest.raises(ValueError):
        embfile._embfile.open(path)
    # providing format_id, it should work
    assert embfile._embfile.open(path, format_id='bin').__class__ == BinaryFile
    assert embfile._embfile.open(path, format_id='txt').__class__ == TextFile


@pytest.mark.parametrize('ext', ['', '.pdf', '.gz', '.zip'])
@patch('embfile._embfile.FORMATS', registry())
def test_open_with_no_useful_extension(ext):
    with pytest.raises(ValueError):
        embfile._embfile.open('path/to/file' + ext)

    assert embfile._embfile.open('path/to/file' + ext, format_id='txt').__class__ == TextFile


@patch('embfile._embfile.FORMATS', registry())
def test_open_with_wrong_format_id():
    with pytest.raises(ValueError):
        embfile._embfile.open('path/to/file.txt', format_id='ciao')


@patch('embfile._embfile.FORMATS', registry())
def test_register_format():
    @embfile.register_format('new', ['.new', '.nw'])
    class NewEmbFile(FakeFile):
        pass

    assert embfile._embfile.open('path/to/file.new').__class__ == NewEmbFile
    assert embfile._embfile.open('path/to/file.nw').__class__ == NewEmbFile


@patch('embfile._embfile.FORMATS', registry())
def test_associate_extension():
    embfile.associate_extension('.text', 'txt')
    assert embfile._embfile.open('path/to/file.text').__class__ == TextFile

    embfile.associate_extension('.binary', 'bin')
    assert embfile._embfile.open('path/to/file.binary').__class__ == BinaryFile

    with pytest.raises(ValueError):
        embfile.associate_extension('.vec', 'vvm')

    embfile.associate_extension('.txt', 'vvm', overwrite=True)
    assert embfile._embfile.open('path/to/file.txt').__class__ == VVMFile


def test_str_FORMATS():
    str(embfile.FORMATS)
