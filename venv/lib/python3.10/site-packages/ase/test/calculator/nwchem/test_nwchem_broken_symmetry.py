"""Check if we can deal with spin-broken symmetries."""
import pytest
from numpy import array

from ase import Atoms


@pytest.mark.calculator('nwchem')
def test_main(factory):
    """Perform C_{\\inf v} calculation on Cr_2."""
    # PBE from
    # J. Chem. Phys. 112 , 5576 (2000)
    # https://doi.org/10.1063/1.481183
    e_literature = -1.34

    names = ['Cr2_sp_up.mos',
             'Cr2_sp_down.mos']
    fragment_energies = array([.0] * 2)
    cr_atom = Atoms('Cr', positions=[(0, 0, 0)], pbc=False)
    for orientation in range(2):    # create two fragments
        imm = 6 * (-1)**orientation
        cr_atom.set_initial_magnetic_moments([imm])
        calc = factory.calc(
            task='energy', xc='pbe', theory='dft',
            dft=dict(convergence=dict(energy=1e-3,
                                      density=1e-2,
                                      gradient=5e-2),
                     vectors='input atomic output {}'
                     .format(names[orientation])),
            charge=0,
            basis='"DZVP2 (DFT Orbital)"'
        )
        cr_atom.calc = calc
        fragment_energies[orientation] = cr_atom.get_potential_energy()
    cr_dimer = Atoms('Cr2', positions=[(0, 0, 0), (0, 0, 1.93)], pbc=False)
    cr_dimer.set_initial_magnetic_moments([0, 0])
    calc = factory.calc(
        task='energy', xc='pbe', theory='dft',
        dft=dict(convergence=dict(energy=1e-3,
                                  density=1e-2,
                                  gradient=5e-2),
                 odft=None,
                 vectors='input fragment {} output Cr2_AF.mos'
                 .format(' '.join(names))),
        basis='"DZVP2 (DFT Orbital)"',
        charge=0
    )
    cr_dimer.calc = calc
    e_dimer = cr_dimer.get_potential_energy()
    e_tot = e_dimer - fragment_energies.sum()
    assert abs(e_tot - e_literature) < 0.01
