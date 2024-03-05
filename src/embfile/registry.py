from typing import Dict, Iterable, List, Type

import tabulate

from embfile.core import EmbFile


class FormatsRegistry:
    """
    Maps each ``EmbFile`` subclass to a format_id and one or multiple file extensions.

    Attributes:
        id_to_class:
        extension_to_id:
        id_to_extensions:
    """
    _TABLEFMT = 'simple'

    def __init__(self) -> None:
        self.id_to_class: Dict[str, Type[EmbFile]] = dict()
        self.extension_to_id: Dict[str, str] = dict()
        self.id_to_extensions: Dict[str, List[str]] = dict()

    def __str__(self) -> str:
        rows = []
        for fid in self.id_to_class:
            classname = self.id_to_class[fid].__name__
            extensions = ', '.join(self.id_to_extensions[fid])
            rows.append((classname, fid, extensions))
        return tabulate.tabulate(rows, headers=('Class', 'Format ID', 'Extensions'),
                                 tablefmt=self._TABLEFMT)

    def register_format(self, embfile_class: Type['EmbFile'],
                        format_id: str,
                        extensions: Iterable[str],
                        overwrite: bool = False) -> None:
        """
        Registers a new embedding file format with a given id and associates the provided
        file extensions to it.

        Args:
            embfile_class:
            format_id:
            extensions:
            overwrite:
        """
        if not overwrite and format_id in self.id_to_class:
            raise ValueError('id {} already registered with class %s' % self.id_to_class[format_id])

        self.id_to_class[format_id] = embfile_class
        self.id_to_extensions[format_id] = []

        for ext in extensions:
            self.associate_extension(ext, format_id, overwrite)

    def associate_extension(
        self, ext: str, format_id: str, overwrite: bool = False
    ) -> None:
        """
        Associates a file extension to a registered embedding file format.

        Args:
            ext:
            format_id:
            overwrite:
        """
        if not ext.startswith('.'):
            raise ValueError('extensions must start with a dot; invalid: ' + ext)

        if format_id not in self.id_to_class:
            raise ValueError('no format {} was registered'.format(format_id))

        if ext in self.extension_to_id:
            if overwrite:
                fid = self.extension_to_id[ext]
                self.id_to_extensions[fid].remove(ext)
            else:
                raise ValueError('the extension {!r} is already registered with format {!r}'
                                 .format(ext, self.extension_to_id[ext]))

        self.extension_to_id[ext] = format_id
        self.id_to_extensions[format_id].append(ext)

    def extensions(self):
        return self.extension_to_id.keys()

    def format_ids(self):
        return self.id_to_class.keys()

    def format_classes(self):
        return self.id_to_class.values()

    def extension_to_class(self, ext):
        return self.id_to_class[self.extension_to_id[ext]]
