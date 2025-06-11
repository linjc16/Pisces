import pytest

from ase.build import molecule


@pytest.mark.calculator_lite()
@pytest.mark.calculator('dftb')
def test_xtb_static(factory):
    atoms = molecule('H2O')
    atoms.calc = factory.calc(
        atoms=atoms, Hamiltonian_='xTB', Hamiltonian_Method='GFN2-xTB'
    )

    e = atoms.get_potential_energy()
    assert e == pytest.approx(-137.9677759924738, rel=0.1)
