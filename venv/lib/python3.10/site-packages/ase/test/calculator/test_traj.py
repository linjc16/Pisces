import pytest

from ase.build import molecule
from ase.io import read, write

calc = pytest.mark.calculator


@calc('aims', sc_accuracy_rho=5.e-3, sc_accuracy_forces=1e-4, xc='LDA',
      kpts=(1, 1, 1))
@calc('gpaw', mode='lcao', basis='sz(dzp)')
@calc('abinit', 'cp2k', 'emt', 'psi4')
@calc('vasp', xc='lda', prec='low')
@calc('crystal', basis='sto-3g')
@calc('gamess_us', label='test_traj')
def test_h2_traj(factory, testdir):
    h2 = molecule('H2')
    h2.center(vacuum=2.0)
    h2.pbc = True
    h2.calc = factory.calc()
    e = h2.get_potential_energy()
    assert not h2.calc.calculation_required(h2, ['energy'])
    f = h2.get_forces()
    assert not h2.calc.calculation_required(h2, ['energy', 'forces'])
    write('h2.traj', h2)
    h2 = read('h2.traj')
    assert abs(e - h2.get_potential_energy()) < 1e-12
    assert abs(f - h2.get_forces()).max() < 1e-12
