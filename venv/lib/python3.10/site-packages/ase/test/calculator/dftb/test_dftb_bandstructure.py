import pytest

from ase.build import bulk
from ase.utils import tokenize_version


@pytest.mark.skip('test is rather broken')
def test_dftb_bandstructure(dftb_factory):
    # We need to get the DFTB+ version to know
    # whether to skip this test or not.
    # For this, we need to run DFTB+ and grep
    # the version from the output header.
    # cmd = os.environ['ASE_DFTB_COMMAND'].split()[0]
    # cmd = dftb_factory.ex
    version = dftb_factory.version()
    if tokenize_version(version) < tokenize_version('17.1'):
        pytest.skip('Band structure requires DFTB 17.1+')

    calc = dftb_factory.calc(
        label='dftb',
        kpts=(3, 3, 3),
        Hamiltonian_SCC='Yes',
        Hamiltonian_SCCTolerance=1e-5,
        Hamiltonian_MaxAngularMomentum_Si='d'
    )

    atoms = bulk('Si')
    atoms.calc = calc
    atoms.get_potential_energy()

    efermi = calc.get_fermi_level()
    assert abs(efermi - -2.90086680996455) < 1.

    # DOS does not currently work because of
    # missing "get_k_point_weights" function
    # from ase.dft.dos import DOS
    # dos = DOS(calc, width=0.2)
    # d = dos.get_dos()
    # e = dos.get_energies()
    # print(d, e)

    calc = dftb_factory.calc(
        atoms=atoms,
        label='dftb',
        kpts={'path': 'WGXWLG', 'npoints': 50},
        Hamiltonian_SCC='Yes',
        Hamiltonian_MaxSCCIterations=1,
        Hamiltonian_ReadInitialCharges='Yes',
        Hamiltonian_MaxAngularMomentum_Si='d'
    )

    atoms.calc = calc
    calc.calculate(atoms)

    # calc.results['fermi_levels'] = [efermi]
    calc.band_structure()
    # Maybe write the band structure or assert something?
