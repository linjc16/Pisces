"""Check that plain cli doesn't execute too many imports."""
import sys

from ase.utils.checkimports import check_imports


def test_imports():
    forbidden_modules = [
        'gpaw',  # external
        'scipy',  # large
        'ase.io.formats',  # possibly slow external formats
        'ase.calculators.(?!names).*',  # any calculator
    ]
    if sys.version_info >= (3, 10):
        max_nonstdlib_module_count = 200  # this depends on the environment
    else:
        max_nonstdlib_module_count = None
    check_imports("from ase.cli.main import main; main(args=[])",
                  forbidden_modules=forbidden_modules,
                  max_nonstdlib_module_count=max_nonstdlib_module_count)
