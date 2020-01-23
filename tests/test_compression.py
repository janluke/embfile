
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from embfile import extract_if_missing
from embfile.compression import (COMPRESSION_TO_EXTENSIONS, extract_file, open_file,
                                 remove_compression_extension, _get_extraction_path)

TEXT_CONTENT = 'Ciao mamma, guarda come mi diverto'


@pytest.mark.parametrize(
    'x, y', (
        ('path/to/file.gz', 'path/to/file'),
        ('path/to/file.txt.bz2', 'path/to/file.txt'),
        ('path/to/file.1M.300.txt.zip', 'path/to/file.1M.300.txt')
    ))
def test_remove_compression_extension(x, y):
    assert remove_compression_extension(x) == Path(y)


def get_file_path(dir_path, filename, compression):
    if compression:
        filename += COMPRESSION_TO_EXTENSIONS[compression][0]
    return Path(dir_path / filename)


def test_get_extraction_path():
    source_path = Path('/path/to/file.txt.zip')

    def f(*args, **kwargs):
        return _get_extraction_path(source_path, *args, **kwargs)

    assert f() == Path('./file.txt')
    assert f(dest_dir=Path('/dir')) == Path('/dir/file.txt')
    assert f(dest_filename='ciao.txt') == Path('./ciao.txt')
    assert f(dest_dir=Path('/dir'), dest_filename='ciao.txt') \
        == Path('/dir/ciao.txt')
    assert f(member='member.txt') == Path('./member.txt')
    assert f(member='member.txt', dest_dir=Path('dir/')) == Path('dir/member.txt')
    assert f(member='member.txt', dest_filename='ciao.txt') == Path('./ciao.txt')
    assert f(member='member.txt', dest_dir=Path('dir/'), dest_filename='ciao.txt') \
        == Path('dir/ciao.txt')


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16'])
@pytest.mark.parametrize('compression', [None, 'zip', 'gz', 'xz'])
def test_open_file(tmp_path, encoding, compression):
    path = get_file_path(tmp_path, 'file.txt', compression)
    # We can't use "wt" mode because of a bug of io.TextIOWrapper with xz and bz2.
    # See: https://stackoverflow.com/questions/55171439/python-bz2-and-lzma-in-mode-wt-dont-write-the-bom-while-gzip-does-why  # noqa
    with open_file(path, 'wb') as f:
        f.write(TEXT_CONTENT.encode(encoding))
    assert path.exists()
    with open_file(path, 'rt', encoding=encoding) as f:
        assert f.read() == TEXT_CONTENT


@pytest.mark.parametrize('compression', list(COMPRESSION_TO_EXTENSIONS))
def test_extract_file_decompress_monofile_archives(tmp_path, compression):
    path = get_file_path(tmp_path, 'file.txt', compression)
    with open_file(path, 'wb') as f:
        f.write(TEXT_CONTENT.encode())

    out_path = extract_file(path, dest_dir=tmp_path)
    text = out_path.read_text()
    assert text == TEXT_CONTENT


def test_extract_file_from_tar(tmp_path):
    # Create a tar file with 2 files
    path = tmp_path / 'file.txt'
    path.write_text(TEXT_CONTENT)
    another_path = Path(tmp_path / 'another.txt')
    another_path.write_text('blah blah blah')
    tar_path = tmp_path / 'archive.tar.gz'
    with tarfile.open(str(tar_path), mode='w:gz') as tar:
        tar.add(str(path), 'file.txt')
        tar.add(str(another_path), 'another.txt')

    out_path = extract_file(tar_path, member='file.txt',
                            dest_dir=tmp_path, dest_filename='extracted.txt')
    assert out_path == tmp_path / 'extracted.txt'
    assert out_path.read_text() == TEXT_CONTENT


def test_extract_file_from_zip(tmp_path):
    # Create a zip file with 2 files
    path = tmp_path / 'file.txt'
    path.write_text(TEXT_CONTENT)
    another_path = Path(tmp_path / 'another.txt')
    another_path.write_text('blah blah blah')
    archive_path = tmp_path / 'archive.tar.gz'
    with zipfile.ZipFile(str(archive_path), mode='w') as archive:
        archive.write(str(path), 'file.txt')
        archive.write(str(another_path), 'another.txt')

    out_path = extract_file(archive_path, member='file.txt',
                            dest_dir=tmp_path, dest_filename='extracted.txt')
    assert out_path == tmp_path / 'extracted.txt'
    assert out_path.read_text() == TEXT_CONTENT


@patch('embfile.compression.extract_file')
def test_extract_if_missing_with_existing_file(extract_file, tmp_path):
    path = tmp_path / 'file.txt'
    compressed_path = tmp_path / 'file.txt.gz'
    path.touch()
    extraction_path = extract_if_missing(compressed_path, dest_dir=tmp_path)
    extract_file.assert_not_called()
    assert extraction_path == path


@patch('embfile.compression.extract_file')
def test_extract_if_missing_with_missing_file(extract_file, tmp_path):
    compressed_path = tmp_path / 'file.txt.gz'
    extract_if_missing(compressed_path, dest_dir=tmp_path)
    extract_file.assert_called_once_with(
        compressed_path, None, tmp_path, None)
