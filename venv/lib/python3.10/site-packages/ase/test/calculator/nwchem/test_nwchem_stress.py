import pytest
from numpy.testing import assert_allclose

from ase.build import bulk
from ase.calculators.fd import calculate_numerical_stress


@pytest.mark.calculator_lite()
@pytest.mark.calculator('nwchem')
def test_main(factory):
    atoms = bulk('C')

    calc = factory.calc(
        theory='pspw',
        label='stress_test',
        nwpw={'lmbfgs': None,
              'tolerances': '1e-9 1e-9'},
    )
    atoms.calc = calc

    assert_allclose(atoms.get_stress(), calculate_numerical_stress(atoms),
                    atol=1e-3, rtol=1e-3)
