import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write


def test_issue276(testdir):
    atoms = bulk("Cu")
    atoms.rattle()
    atoms.calc = EMT()
    forces = atoms.get_forces()

    write("tmp.xyz", atoms)
    atoms2 = read("tmp.xyz")
    forces2 = atoms.get_forces()

    assert np.abs(forces - forces2).max() < 1e-6

    write("tmp2.xyz", atoms2)
    atoms3 = read("tmp2.xyz")
    forces3 = atoms3.get_forces()
    assert np.abs(forces - forces3).max() < 1e-6
