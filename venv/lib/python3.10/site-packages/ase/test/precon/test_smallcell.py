import pytest

from ase.atoms import Atoms
from ase.optimize.precon import PreconLBFGS


def test_precon_warn():
    with pytest.warns(UserWarning, match='The system is likely too small'):
        PreconLBFGS(Atoms('H'))


def test_precon_nowarn():
    PreconLBFGS(Atoms('100H'))
