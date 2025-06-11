# fmt: off

import importlib
from typing import List, Tuple

from ase.utils import (
    get_python_package_path_description,
    search_current_git_hash,
)


def format_dependency(modname: str) -> Tuple[str, str]:
    """Return (name, info) for given module.

    If possible, info is the path to the module's package."""
    try:
        module = importlib.import_module(modname)
    except ImportError:
        return modname, 'not installed'

    version = getattr(module, '__version__', '?')
    name = f'{modname}-{version}'
    if modname == 'ase':
        githash = search_current_git_hash(module)
        if githash:
            name += f'-{githash:.10}'

    # (only packages have __path__, but we are importing packages.)
    info = get_python_package_path_description(module)
    return name, info


def all_dependencies() -> List[Tuple[str, str]]:
    names = ['ase', 'numpy', 'scipy', 'matplotlib', 'spglib',
             'ase_ext', 'flask', 'psycopg2', 'pyamg']
    return [format_dependency(name) for name in names]
