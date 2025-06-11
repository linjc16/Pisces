import numpy as np

from ase.dft.kpoints import bandpath


def test_kpts():
    print(bandpath('GX,GX', np.eye(3), 6))
