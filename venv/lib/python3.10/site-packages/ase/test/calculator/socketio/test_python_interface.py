import os

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.socketio import PySocketIOClient, SocketIOCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS


@pytest.mark.optimize()
@pytest.mark.skipif(os.name != 'posix', reason='only posix')
def test_socketio_python():

    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.05)
    fmax = 0.01
    atoms.cell += np.random.RandomState(42).rand(3, 3) * 0.05

    client = PySocketIOClient(EMT)

    pid = os.getpid()
    with SocketIOCalculator(launch_client=client,
                            unixsocket=f'ase-python-{pid}') as atoms.calc:
        with BFGS(FrechetCellFilter(atoms)) as opt:
            opt.run(fmax=fmax)

    residual_forces = np.linalg.norm(atoms.get_forces(), axis=1)
    assert len(residual_forces) == len(atoms)
    assert np.max(residual_forces) < fmax
