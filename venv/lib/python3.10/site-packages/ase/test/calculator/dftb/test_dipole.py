import numpy as np
import pytest

from ase.build import molecule


@pytest.mark.calculator_lite()
@pytest.mark.calculator('dftb')
def test_H2O(factory):
    atoms = molecule('H2O')
    atoms.calc = factory.calc(label='h2o')

    dipole = atoms.get_dipole_moment()

    assert np.linalg.norm(dipole) == pytest.approx(0.35212409930846317)
