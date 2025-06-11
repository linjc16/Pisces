import pytest

from ase.build import molecule
from ase.utils import workdir

calc = pytest.mark.calculator
filterwarnings = pytest.mark.filterwarnings


def _calculate(code, name):
    atoms = molecule(name)
    atoms.center(vacuum=3.5)
    with workdir(f'test-{name}', mkdir=True):
        atoms.calc = code.calc()
        return atoms.get_potential_energy()


omx_par = {'definition_of_atomic_species': [['C', 'C6.0', 'C_CA19'],
                                            ['H', 'H5.0', 'H_CA19']]}


@pytest.mark.calculator_lite()
@calc('abinit', ecut=300, chksymbreak=0, toldfe=1e-4)
@calc('aims')
@calc('cp2k')
@calc('espresso')
@calc('gpaw', symmetry='off', mode='pw', txt='gpaw.txt', mixer={'beta': 0.6},
      marks=[filterwarnings('ignore:.*?ignore_bad_restart_file'),
             filterwarnings('ignore:convert_string_to_fd')])
@calc('nwchem')
@calc('octopus', Spacing='0.25 * angstrom', BoxShape='minimum',
      convreldens=1e-3, Radius='3.5 * angstrom')
@calc('openmx', **omx_par)
@calc('siesta', marks=pytest.mark.xfail)
@calc('gamess_us', label='ch4')
@calc('gaussian', xc='lda', basis='3-21G')
def test_ch4_reaction(factory):
    e_ch4 = _calculate(factory, 'CH4')
    e_c2h2 = _calculate(factory, 'C2H2')
    e_h2 = _calculate(factory, 'H2')
    energy = e_ch4 - 0.5 * e_c2h2 - 1.5 * e_h2
    print(energy)
    ref_energy = -2.8
    assert abs(energy - ref_energy) < 0.3
