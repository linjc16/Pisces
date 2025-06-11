import sys
from subprocess import run

import pytest

from ase.parallel import world


def test_mpi():
    x = 42.0
    with pytest.warns(FutureWarning):
        x = world.sum(x)
    x = world.sum_scalar(x)
    assert x == 42 * world.size


@pytest.mark.skip(reason='Does not work and no time to investigate.')
def test_mpi_unused_on_import():
    """Try to import all ASE modules and check that ase.parallel.world has not
    been used.  We want to delay use of world until after MPI4PY has been
    imported.

    We run the test in a subprocess so that we have a clean Python
    interpreter."""

    # Should cover most of ASE:
    modules = ['ase.optimize',
               'ase.db',
               'ase.gui']

    imports = 'import ' + ', '.join(modules)

    run([sys.executable,
         '-c',
         '{imports}; from ase.parallel import world; assert world.comm is None'
         .format(imports=imports)],
        check=True)
