import pytest

from ase import io
from ase.build import molecule


@pytest.mark.calculator('gpaw')
@pytest.mark.filterwarnings('ignore:The keyword')
@pytest.mark.filterwarnings('ignore:convert_string_to_fd')
# Ignore calculator constructor keyword warning for now
def test_no_spin_and_spin(factory):
    txt = 'out.txt'

    calculator = factory.calc(mode='fd', h=0.3, txt=txt)
    atoms = molecule('H2', calculator=calculator)
    atoms.center(vacuum=3)
    atoms.get_potential_energy()
    atoms.set_initial_magnetic_moments([0.5, 0.5])
    calculator.set(charge=1)
    atoms.get_potential_energy()

    # read again
    t = io.read(txt, index=':')
    assert isinstance(t, list)
    M = t[1].get_magnetic_moments()
    assert abs(M - 0.2).max() < 0.1
