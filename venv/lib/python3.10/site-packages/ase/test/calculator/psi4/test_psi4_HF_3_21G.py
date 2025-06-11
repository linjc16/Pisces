import pytest
from numpy.testing import assert_allclose

from ase.build import molecule
from ase.calculators.fd import calculate_numerical_forces


@pytest.mark.skip(reason='we do not know how to compile psi4')
@pytest.mark.filterwarnings('once::DeprecationWarning')
@pytest.mark.calculator_lite()
@pytest.mark.calculator('psi4')
def test_main(factory):
    atoms = molecule('H2O')
    atoms.rotate(30, 'x')

    calc = factory.calc(basis='3-21G')
    atoms.calc = calc

    # Calculate forces ahead of time, compare against finite difference after
    # checking the psi4-calc.dat file
    atoms.get_forces()
    assert_allclose(atoms.get_potential_energy(), -2056.785854116688,
                    rtol=1e-4, atol=1e-4)

    # Test the reader
    calc2 = factory.calc()
    calc2.read('psi4-calc')
    assert_allclose(calc2.results['energy'], atoms.get_potential_energy(),
                    rtol=1e-4, atol=1e-4)
    assert_allclose(calc2.results['forces'], atoms.get_forces(),
                    rtol=1e-4, atol=1e-4)

    # Compare analytical vs numerical forces
    assert_allclose(atoms.get_forces(), calculate_numerical_forces(atoms),
                    rtol=1e-4, atol=1e-4)
