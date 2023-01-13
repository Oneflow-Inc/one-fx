"""
Modified from https://github.com/pytorch/pytorch/blob/992dad18552883224016d53429c48f2e932651cf/torch/package/package_exporter.py
"""

import collections
import importlib.machinery
import io
import linecache
import pickletools
import platform
import types
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    cast,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import Storage
from torch.utils.hooks import RemovableHandle

from ._mangling import demangle, is_mangled
from .importer import Importer, OrderedImporter, sys_importer

__all__ = [
    "PackagingErrorReason",
    "EmptyMatchError",
    "PackagingError",
    "PackageExporter",
]


class PackageExporter:
    """Exporters allow you to write packages of code, pickled Python data, and
    arbitrary binary and text resources into a self-contained package.

    Imports can load this code in a hermetic way, such that code is loaded
    from the package rather than the normal Python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The code contained in packages is copied file-by-file from the original
    source when it is created, and the file format is a specially organized
    zip file. Future users of the package can unzip the package, and edit the code
    in order to perform custom modifications to it.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external using :meth:`extern`.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.

    When source code is added to the package, the exporter can optionally scan it
    for further code dependencies (``dependencies=True``). It looks for import statements,
    resolves relative references to qualified module names, and performs an action specified by the user
    (See: :meth:`extern`, :meth:`mock`, and :meth:`intern`).
    """

    """A importer that will be searched in order to find the modules referenced by other modules or by
    pickled objects. The default module environment just uses sys_importer, which searches the Python environment.
    """
    importer: Importer

    def __init__(
        self,
        f: Union[str, Path, BinaryIO],
        importer: Union[Importer, Sequence[Importer]] = sys_importer,
    ):
        pass
