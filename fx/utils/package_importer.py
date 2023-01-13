"""
Modified from https://github.com/pytorch/pytorch/blob/992dad18552883224016d53429c48f2e932651cf/torch/package/package_importer.py
"""

import builtins
import importlib
import importlib.machinery
import inspect
import io
import linecache
import os.path
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, cast, Dict, Iterable, List, Optional, Union
from weakref import WeakValueDictionary

from ._mangling import demangle, PackageMangler
from .importer import Importer

__all__ = ["PackageImporter"]

# TODO: Maybe delete it.

# This is a list of imports that are implicitly allowed even if they haven't
# been marked as extern. This is to work around the fact that Torch implicitly
# depends on numpy and package can't track it.
IMPLICIT_IMPORT_ALLOWLIST: Iterable[str] = [
    "numpy",
    "numpy.core",
    "numpy.core._multiarray_umath",
    # FX GraphModule might depend on builtins module and users usually
    # don't extern builtins. Here we import it here by default.
    "builtins",
]


class PackageImporter(Importer):
    """Importers allow you to load code written to packages by :class:`PackageExporter`.
    Code is loaded in a hermetic way, using files from the package
    rather than the normal python import system. This allows
    for the packaging of Oneflow model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external during export.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.
    """

    """The dictionary of already loaded modules from this package, equivalent to ``sys.modules`` but
    local to this importer.
    """

    modules: Dict[str, types.ModuleType]

    def __init__(
        self,
        file_or_buffer: Union[str, Path],
        module_allowed: Callable[[str], bool] = lambda module_name: True,
    ):
        pass

    def import_module(self, name: str, package=None):
        """Load a module from the package if it hasn't already been loaded, and then return
        the module. Modules are loaded locally
        to the importer and will appear in ``self.modules`` rather than ``sys.modules``.

        Args:
            name (str): Fully qualified name of the module to load.
            package ([type], optional): Unused, but present to match the signature of importlib.import_module. Defaults to ``None``.

        Returns:
            types.ModuleType: The (possibly already) loaded module.
        """
        # We should always be able to support importing modules from this package.
        # This is to support something like:
        #   obj = importer.load_pickle(...)
        #   importer.import_module(obj.__module__)  <- this string will be mangled
        #
        # Note that _mangler.demangle will not demangle any module names
        # produced by a different PackageImporter instance.

        return ""
