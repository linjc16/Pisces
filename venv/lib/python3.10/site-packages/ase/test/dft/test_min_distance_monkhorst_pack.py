import numpy as np

from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack as kd2mp
from ase.dft.kpoints import mindistance2monkhorstpack as md2mp


def test_non_periodic():
    assert np.prod(md2mp(Atoms(cell=(1, 1, 1), pbc=False),
                         min_distance=4)) == 1


def test_orthogonal():
    for b in range(8):
        atoms = Atoms(cell=(2, 3, 4),
                      pbc=[b | 0x1 == b, b | 0x2 == b, b | 0x4 == b])
        assert np.all(md2mp(atoms, min_distance=4 * 2 * np.pi) ==
                      kd2mp(atoms, kptdensity=4.0))


def test_tricky():
    atoms = Atoms(cell=[[1.0, 2.0, 3.0], [-1.0, 2.0, 3.0], [3.0, -2.0, 2.0]],
                  pbc=True)
    assert np.all(md2mp(atoms, min_distance=4 * 2 * np.pi) == [16, 6, 8])
    assert np.all(kd2mp(atoms, kptdensity=4.0) == [20, 20, 10])
