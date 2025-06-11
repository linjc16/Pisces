import numpy as np
import pytest

from ase.build import molecule


@pytest.fixture()
def atoms():
    return molecule('H2')


@pytest.mark.calculator('nwchem')
@pytest.mark.parametrize('charge, eref', ((-1, -24.036791014064605),
                                          (1, -14.365500960943171)))
def test_nwchem_eigenvalues(factory, atoms, charge, eref):
    atoms.calc = factory.calc(charge=charge, dft=dict(mult=2))
    energy = atoms.get_potential_energy()
    assert abs(energy - eref) < 0.1

    # Test fix for issue 575, which caused positive eigenvalues to not parse
    # correctly. Make sure at least some of the eigenvalues are positive.
    # (Actually they all should be positive, but let's be less strict)
    evals = atoms.calc.calc.get_eigenvalues()
    assert np.any(evals > 0)
    assert len(evals) == 4
