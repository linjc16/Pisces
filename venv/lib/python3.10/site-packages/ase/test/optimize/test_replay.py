from math import sqrt

import pytest

from ase import Atom, Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import QuasiNewton


@pytest.mark.optimize()
def test_replay(testdir):
    # Distance between Cu atoms on a (100) surface:
    d = 3.6 / sqrt(2)
    a = Atoms('Cu',
              positions=[(0, 0, 0)],
              cell=(d, d, 1.0),
              pbc=(True, True, False))
    a *= (2, 2, 1)  # 2x2 (100) surface-cell

    # Approximate height of Ag atom on Cu(100) surfece:
    h0 = 2.0
    a += Atom('Ag', (d / 2, d / 2, h0))

    constraint = FixAtoms(range(len(a) - 1))
    a.calc = EMT()
    a.set_constraint(constraint)

    with QuasiNewton(a, trajectory='AgCu1.traj', logfile='AgCu1.log') as dyn1:
        dyn1.run(fmax=0.1)

    a = read('AgCu1.traj')
    a.calc = EMT()
    print(a.constraints)

    with QuasiNewton(a, trajectory='AgCu2.traj', logfile='AgCu2.log') as dyn2:
        dyn2.replay_trajectory('AgCu1.traj')
        dyn2.run(fmax=0.01)
