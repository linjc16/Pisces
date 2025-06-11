import pytest

from ase import Atoms
from ase.calculators.morse import MorsePotential
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.optimize import QuasiNewton


@pytest.mark.optimize()
def test_example(testdir):
    atoms = Atoms('H7',
                  positions=[(0, 0, 0),
                             (1, 0, 0),
                             (0, 1, 0),
                             (1, 1, 0),
                             (0, 2, 0),
                             (1, 2, 0),
                             (0.5, 0.5, 1)],
                  constraint=[FixAtoms(range(6))],
                  calculator=MorsePotential())

    with Trajectory('H.traj', 'w', atoms) as traj, \
            QuasiNewton(atoms, maxstep=0.2) as dyn:
        dyn.attach(traj.write)
        dyn.run(fmax=0.01, steps=100)

    print(atoms)
    del atoms[-1]
    print(atoms)
    del atoms[5]
    print(atoms)
    assert len(atoms.constraints[0].index) == 5
