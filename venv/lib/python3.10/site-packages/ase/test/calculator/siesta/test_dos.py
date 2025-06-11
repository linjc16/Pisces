import pytest

from ase.build import bulk
from ase.dft import DOS


@pytest.mark.calculator_lite()
@pytest.mark.calculator('siesta')
def test_dos(factory):
    atoms = bulk('Si')
    atoms.calc = factory.calc(kpts=[2, 2, 2])
    atoms.get_potential_energy()
    DOS(atoms.calc, width=0.2)
