import numpy as np
import pytest

from ase.build import bulk, fcc111
from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)


@pytest.mark.calculator('eam')
@pytest.mark.calculator_lite()
def test_eam_run(factory):
    with open(f'{factory.factory.potentials_path}/Pt_u3.eam') as fd:
        eam = factory.calc(potential=fd, form='eam', elements=['Pt'])
    slab = fcc111('Pt', size=(4, 4, 2), vacuum=10.0)
    slab.calc = eam

    assert abs(-164.277599313 - slab.get_potential_energy()) < 1E-8
    assert abs(6.36379627645 - np.linalg.norm(slab.get_forces())) < 1E-8


@pytest.mark.parametrize('with_elements', (False, True))
@pytest.mark.parametrize(
    'potential',
    ('Pt_u3.eam', 'NiAlH_jea.eam.alloy', 'NiAlH_jea.eam.fs', 'AlCu.adp'),
)
@pytest.mark.calculator('eam')
@pytest.mark.calculator_lite()
def test_read_potential(factory, potential: str, with_elements: bool):
    """Test if the potential can be read without errors."""
    element = potential[:2]
    potential = f'{factory.factory.potentials_path}/{potential}'
    if with_elements:
        calc = factory.calc(potential=potential, elements=[element])
    else:
        calc = factory.calc(potential=potential)
    atoms = bulk(element)
    atoms.calc = calc
    atoms.get_potential_energy()

    # test forces against numerical forces
    forces = atoms.get_forces()
    numerical_forces = calculate_numerical_forces(atoms, eps=1e-5)
    np.testing.assert_allclose(forces, numerical_forces, atol=1e-5)

    # test stress against numerical stress
    stress = atoms.get_stress()
    numerical_stress = calculate_numerical_stress(atoms, eps=1e-5)
    np.testing.assert_allclose(stress, numerical_stress, atol=1e-5)
