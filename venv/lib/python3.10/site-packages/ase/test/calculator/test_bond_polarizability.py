import numpy as np
import pytest

from ase import Atoms
from ase.calculators.bond_polarizability import (
    BondPolarizability,
    Linearized,
    LippincottStuttman,
)


def test_CC_bond():
    """Test polarizabilties of a single CC bond"""
    C2 = Atoms('C2', positions=[[0, 0, 0], [0, 0, 1.69]])

    def check_symmetry(alpha):
        alpha_diag = np.diagonal(alpha)
        assert alpha == pytest.approx(np.diag(alpha_diag))
        assert alpha_diag[0] == alpha_diag[1]

    bp = BondPolarizability()
    check_symmetry(bp(C2))
    bp = BondPolarizability(Linearized())
    check_symmetry(bp(C2))


def test_symmetry():
    for lin in [LippincottStuttman(), Linearized()]:
        assert lin('B', 'N', 1) == lin('N', 'B', 1)


def test_2to3():
    """Compare polarizabilties of one and two bonds"""
    Si2 = Atoms('Si2', positions=[[0, 0, 0], [0, 0, 2.5]])
    Si3 = Atoms('Si3', positions=[[0, 0, -2.5], [0, 0, 0], [0, 0, 2.5]])
    bp = BondPolarizability()
    bp2 = bp(Si2)
    # polarizability is a tensor
    assert bp2.shape == (3, 3)
    # check sum of equal bonds
    assert bp(Si3) == pytest.approx(2 * bp2)
