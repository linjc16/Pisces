# fmt: off

"""Utility for checking Python module imports triggered by any code snippet.

This module was developed to monitor the import footprint of the ase CLI
command: The CLI command can become unnecessarily slow and unresponsive
if too many modules are imported even before the CLI is launched or
it is known what modules will be actually needed.
See https://gitlab.com/ase/ase/-/issues/1124 for more discussion.

The utility here is general, so it can be used for checking and
monitoring other code snippets too.
"""
import json
import os
import re
import sys
from pprint import pprint
from subprocess import run
from typing import List, Optional, Set


def exec_and_check_modules(expression: str) -> Set[str]:
    """Return modules loaded by the execution of a Python expression.

    Parameters
    ----------
    expression
        Python expression

    Returns
    -------
    Set of module names.
    """
    # Take null outside command to avoid
    # `import os` before expression
    null = os.devnull
    command = ("import sys;"
               f" stdout = sys.stdout; sys.stdout = open({null!r}, 'w');"
               f" {expression};"
               " sys.stdout = stdout;"
               " modules = list(sys.modules);"
               " import json; print(json.dumps(modules))")
    proc = run([sys.executable, '-c', command],
               capture_output=True, universal_newlines=True,
               check=True)
    return set(json.loads(proc.stdout))


def check_imports(expression: str, *,
                  forbidden_modules: List[str] = [],
                  max_module_count: Optional[int] = None,
                  max_nonstdlib_module_count: Optional[int] = None,
                  do_print: bool = False) -> None:
    """Check modules imported by the execution of a Python expression.

    Parameters
    ----------
    expression
        Python expression
    forbidden_modules
        Throws an error if any module in this list was loaded.
    max_module_count
        Throws an error if the number of modules exceeds this value.
    max_nonstdlib_module_count
        Throws an error if the number of non-stdlib modules exceeds this value.
    do_print:
        Print loaded modules if set.
    """
    modules = exec_and_check_modules(expression)

    if do_print:
        print('all modules:')
        pprint(sorted(modules))

    for module_pattern in forbidden_modules:
        r = re.compile(module_pattern)
        for module in modules:
            assert not r.fullmatch(module), \
                f'{module} was imported'

    if max_nonstdlib_module_count is not None:
        assert sys.version_info >= (3, 10), 'Python 3.10+ required'

        nonstdlib_modules = []
        for module in modules:
            if (
                module.split('.')[0]
                in sys.stdlib_module_names  # type: ignore[attr-defined]
            ):
                continue
            nonstdlib_modules.append(module)

        if do_print:
            print('nonstdlib modules:')
            pprint(sorted(nonstdlib_modules))

        module_count = len(nonstdlib_modules)
        assert module_count <= max_nonstdlib_module_count, (
            'too many nonstdlib modules loaded:'
            f' {module_count}/{max_nonstdlib_module_count}'
        )

    if max_module_count is not None:
        module_count = len(modules)
        assert module_count <= max_module_count, \
            f'too many modules loaded: {module_count}/{max_module_count}'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('expression')
    parser.add_argument('--forbidden_modules', nargs='+', default=[])
    parser.add_argument('--max_module_count', type=int, default=None)
    parser.add_argument('--max_nonstdlib_module_count', type=int, default=None)
    parser.add_argument('--do_print', action='store_true')
    args = parser.parse_args()

    check_imports(**vars(args))
