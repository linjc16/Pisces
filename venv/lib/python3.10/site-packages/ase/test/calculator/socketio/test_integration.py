import numpy as np
import pytest

from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.units import Ry


# XXX 2023-08-07: segfaults with ecutwfc=300 / Ry and espresso-7.0 (Ubuntu)
@pytest.mark.calculator_lite()
@pytest.mark.calculator('espresso',
                        input_data={"system": {"ecutwfc": 200 / Ry}})
@pytest.mark.calculator('abinit')
def test_socketio_espresso(factory):
    atoms = bulk('Si')
    atoms.rattle(stdev=.2, seed=42)

    with BFGS(FrechetCellFilter(atoms)) as opt, \
            pytest.warns(UserWarning, match='Subprocess exited'), \
            factory.socketio(unixsocket=f'ase_test_socketio_{factory.name}',
                             kpts=[2, 2, 2]) as atoms.calc:

        for _ in opt.irun(fmax=0.05):
            e = atoms.get_potential_energy()
            fmax = max(np.linalg.norm(atoms.get_forces(), axis=0))
            print(e, fmax)
