"""Use C-extensions from asecext.

This module defines a decorator that can be used to replace pure Python
functions with faster C-implementations from the ase_ext module.
"""

import functools

try:
    import ase_ext
except ImportError:
    ase_ext = None


def cextension(func):
    if ase_ext is None:
        return func
    cfunc = getattr(ase_ext, func.__name__, None)
    if cfunc is None:
        return func
    functools.update_wrapper(cfunc, func)
    cfunc.__pure_python_function__ = func
    return cfunc
