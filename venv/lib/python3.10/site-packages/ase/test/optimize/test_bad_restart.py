import pytest

from ase import Atoms
from ase.optimize import BFGS, RestartError


@pytest.mark.optimize()
def test_bad_restart(testdir):
    fname = 'tmp.dat'

    with open(fname, 'w') as fd:
        fd.write('hello world\n')

    with pytest.raises(RestartError, match='Could not decode'):
        BFGS(Atoms(), restart=fname)
