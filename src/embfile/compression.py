__all__ = [
    'open_file', 'extract_file', 'extract_if_missing',
    'EXTENSION_TO_COMPRESSION', 'COMPRESSION_TO_EXTENSIONS',
]

import bz2
import gzip
import io
import logging
import lzma
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from tarfile import TarFile, TarInfo
from typing import Callable, Optional, Sequence, TypeVar, Union
from zipfile import ZipFile

from embfile._utils import invert_one_to_many
from embfile.types import PathType

logger = logging.getLogger(__name__)


class _ZippedFile:
    """ A member file of a zip archive. It's an adapter class that mimics the interface of the
    objects returned by modules io, gzip, lzma, and bz2. It also has a static method ``open``
    that mimics the ``open`` method of the above modules.

    In read mode, it can open a specific member file providing ``member``.

    In write mode, it supports only zip archive containing a single file:

    - in mode 'w', it overwrite the archive if it already exists
    - in mode 'x', it raises an error if the archive already exists
    """

    def __init__(self, path: PathType,
                 mode: str = 'r',
                 member: Optional[str] = None,
                 encoding: Optional[str] = None) -> None:

        self.path = Path(path)
        self.mode = mode
        self.zip_mode = mode[0]

        tmp_dir = Path(tempfile.mkdtemp(prefix='embfile_'))

        if self.zip_mode == 'r':
            self._tmp_path = extract_file(src_path=path, member=member, dest_dir=tmp_dir,
                                          dest_filename='embfile_reading.tmp')
            self._tmp_file = open(self._tmp_path, mode, encoding=encoding)

        elif self.zip_mode in ('w', 'x'):
            self._zip_file = zipfile.ZipFile(str(self.path), mode=self.zip_mode)  # type: ignore
            self._tmp_path = tmp_dir / 'embfile_writing.tmp'
            self._tmp_file = open(self._tmp_path, mode, encoding=encoding)
            if member is None:
                member = str(remove_compression_extension(self.path.name))
            self.member = member

        else:
            raise ValueError('unknown mode: ' + mode)

    @classmethod
    def open(cls, path, mode='r', encoding=None):
        return cls(path, mode=mode, encoding=encoding)

    def close(self):
        self._tmp_file.close()
        if self.zip_mode in ('w', 'x'):
            self._zip_file.write(self._tmp_path, arcname=self.member)
            self._zip_file.close()
        os.remove(self._tmp_path)

    def __enter__(self):
        return self._tmp_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return iter(self._tmp_file)

    def __next__(self):
        return next(self._tmp_file)

    def readline(self):
        return self._tmp_file.readline()


#: Given the normalized form of a valid ``compression`` argument, returns its aliases
COMPRESSION_TO_ALIASES = {
    'gz': {'gz', 'gzip'},
    'bz2': {'bz2'},
    'xz': {'xz', 'lzma'},
    'zip': {'zip'}
}

#: Given a valid value for the ``compression`` argument, returns its normalized form
ALIAS_TO_COMPRESSION = invert_one_to_many(COMPRESSION_TO_ALIASES)

#: Maps each compression format to its associated extensions
COMPRESSION_TO_EXTENSIONS = {
    'gz': ['.gz', '.gzip'],
    'bz2': ['.bz2'],
    'xz': ['.xz', '.lzma'],
    'zip': ['.zip']
}

#: Maps a compression extensions to the corresponding compression format name
EXTENSION_TO_COMPRESSION = invert_one_to_many(COMPRESSION_TO_EXTENSIONS)

_COMPRESSION_OPENER = {
    'gz': gzip,
    'bz2': bz2,
    'xz': lzma,
    'zip': _ZippedFile
}


def normalize_compression_arg(compression):
    if compression in ALIAS_TO_COMPRESSION:
        return ALIAS_TO_COMPRESSION[compression]
    raise ValueError('unknown compression "%s"; known compressions are: %s'
                     % ', '.join(_COMPRESSION_OPENER.keys()))


def remove_compression_extension(path: PathType) -> Path:
    path = Path(path)
    if path.suffix in EXTENSION_TO_COMPRESSION:
        return path.parent / path.stem
    return path


def default_compression_ext(compression: str) -> str:
    return COMPRESSION_TO_EXTENSIONS[compression][0]


def open_file(path: PathType, mode: str = 'rt',
              encoding: Optional[str] = None,
              compression: Optional[str] = None):
    """
    Open a file, eventually with (de)compression.

    If ``compression`` is not given, it is inferred from the file extension. If the file has not the
    extension of a supported compression format, the file is opened without compression, unless the
    argument ``compression`` is given.
    """
    path = Path(path)

    if compression:
        compression = normalize_compression_arg(compression)
    else:
        compression = EXTENSION_TO_COMPRESSION.get(path.suffix)

    opener = (io if not compression
              else _COMPRESSION_OPENER[compression])

    return opener.open(path, mode, encoding=encoding)  # type: ignore


A = TypeVar('A', TarFile, ZipFile)


def _archive_member_extractor(
    open_archive: Callable[[PathType], A],
    get_archive_members: Callable[[A], Sequence[str]],
    extract_member: Callable[[A, str, PathType], PathType]
):
    """ Returns a function that extracts a member file from an archive that can contain
    multiple files (zip, tar). """

    def extract_archive_member(
        archive_path: Path,
        member: Optional[str] = None,
        dest_dir: Path = Path('.'),
        dest_filename: Optional[str] = None,
    ) -> Path:
        with open_archive(archive_path) as archive:
            if not member:
                members_list = get_archive_members(archive)
                if len(members_list) == 0:
                    raise ValueError('empty archive: %s' % archive_path)
                if len(members_list) > 1:
                    raise ValueError(
                        'the archive %s contains more than one file, you must provide the '
                        'argument "member"' % archive_path)
                member = members_list[0]
            path = extract_member(archive, member, dest_dir)

        if dest_filename:
            dest_path = dest_dir / dest_filename
            os.rename(path, dest_path)
            return dest_path
        else:
            return path

    return extract_archive_member


_extract_zip_member = _archive_member_extractor(
    open_archive=ZipFile,
    get_archive_members=lambda arc: arc.namelist(),  # type: ignore
    extract_member=lambda arc, member, outdir: arc.extract(member, outdir)  # type: ignore
)


def _tar_extract(archive: TarFile, member: Union[str, TarInfo], outdir: PathType) -> Path:
    archive.extract(member, outdir)
    member_name = member.name if isinstance(member, TarInfo) else member
    return Path(outdir) / member_name


_extract_tar_member = _archive_member_extractor(
    open_archive=tarfile.open,
    get_archive_members=lambda arc: arc.members,  # type: ignore
    extract_member=_tar_extract
)


def _decompress_file(src_path, member, dest_dir, dest_filename):
    """ Suitable for monofile archives. ``member`` is not used (Adapter). """
    dest_path = dest_dir / dest_filename
    with open_file(src_path, 'rb') as fin, open(dest_path, 'wb') as fout:
        shutil.copyfileobj(fin, fout)


def _get_extraction_path(src_path: Path,
                         member: Optional[str] = None,
                         dest_dir: Path = Path('.'),
                         dest_filename: Optional[str] = None) -> Path:
    if not dest_filename:
        dest_filename = member or str(remove_compression_extension(src_path.name))
    return dest_dir / dest_filename


def _is_tarfile_but_not_vvm(path: Path):
    return '.vvm' not in path.suffixes[-2:] and tarfile.is_tarfile(str(path))


def extract_file(src_path: PathType,
                 member: Optional[str] = None,
                 dest_dir: PathType = '.',
                 dest_filename: Optional[str] = None,
                 overwrite: bool = False) -> Path:
    """
    Extracts a file compressed with gzip, bz2 or lzma or a member file inside a
    zip/tar archive. The compression format is inferred from the extension or from
    the magic number of the file (in the case of zip and tar).

    The file is first extracted to a .part file that is renamed when the extraction
    is completed.

    Args:
        src_path:
            source file path
        member:
            must be provided if src_path points to an archive that contains multiple files;
        dest_dir:
            destination directory; by default, it's the current working directory
        dest_filename:
            destination filename; by default, it's equal to ``member`` (if provided)
        overwrite:
            overwrite existing file at ``dest_path`` if it already exists

    Returns:
        Path: the path to the extracted file
    """
    src_path = Path(src_path)
    dest_path = _get_extraction_path(src_path, member, Path(dest_dir), dest_filename)

    if dest_path.exists():
        if overwrite:
            logger.info('The file %s already exists and overwrite is True. '
                        'Removing the file', dest_path)
            dest_path.unlink()
        else:
            raise FileExistsError(dest_path)

    # Select the proper extractor
    if zipfile.is_zipfile(src_path):
        extractor = _extract_zip_member
    elif _is_tarfile_but_not_vvm(src_path):  # <-- TODO: This is ugly
        extractor = _extract_tar_member
    else:
        extractor = _decompress_file

    # Extract to .part file
    temp_filename = dest_path.name + '.part'
    extractor(src_path, member, dest_dir, temp_filename)

    # Remove the .part extension from the file
    temp_dest_path = dest_path.with_name(temp_filename)
    temp_dest_path.rename(dest_path)
    return dest_path


def extract_if_missing(src_path: PathType,
                       member: Optional[str] = None,
                       dest_dir: PathType = '.',
                       dest_filename: Optional[str] = None) -> Path:
    """
    Extracts a file unless it already exists and returns its path.

    Note: during extraction, a .part file is used, so there's no risk of using
    a partially extracted file.

    Args:
        src_path:
        member:
        dest_dir:
        dest_filename:

    Returns:
        The path of the decompressed file is returned.
    """
    dest_path = _get_extraction_path(Path(src_path), member, Path(dest_dir), dest_filename)
    if dest_path.exists():
        logger.info("Nothing to decompress, the file already exists: %s", dest_path)
        return dest_path
    else:
        logger.info("Extracting file to: %s", dest_path)
        return extract_file(src_path, member, dest_dir, dest_filename)
