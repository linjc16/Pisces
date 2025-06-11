from pytest import approx, fixture

from ase import Atoms
from ase.build import bulk
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
from ase.vibrations.placzek import PlaczekStatic, PlaczekStaticPhonons
from ase.vibrations.raman import (
    StaticRamanCalculator,
    StaticRamanPhononsCalculator,
)


@fixture(scope='module')
def Cbulk():
    Cbulk = bulk('C', crystalstructure='fcc', a=2 * 1.221791471)
    Cbulk = Cbulk.repeat([2, 1, 1])
    Cbulk.calc = EMT()
    return Cbulk


def test_bulk(Cbulk, testdir):
    """Bulk FCC carbon (for EMT) self consistency"""
    delta = 0.02

    name = 'bp'
    rm = StaticRamanCalculator(Cbulk, BondPolarizability, name=name,
                               delta=delta)
    rm.run()

    pz = PlaczekStatic(Cbulk, name=name)
    e_vib = pz.get_energies()
    i_vib = pz.get_absolute_intensities()
    assert len(e_vib) == 6
    pz.summary()

    name = 'phbp'
    rm = StaticRamanPhononsCalculator(Cbulk, BondPolarizability,
                                      calc=EMT(),
                                      name=name,
                                      delta=delta, supercell=(1, 1, 1))
    rm.run()

    pz = PlaczekStaticPhonons(Cbulk, name=name)
    e_phonons = pz.get_energies()
    assert len(e_vib) == len(e_phonons)
    assert e_vib[3:] == approx(e_phonons[3:], 1)
    i_phonons = pz.get_absolute_intensities()
    assert i_vib[3:] == approx(i_phonons[3:], 1)
    pz.summary()

    assert rm.clean() == 26


def test_bulk_kpts(Cbulk, testdir):
    """Bulk FCC carbon (for EMT) for phonons"""

    name = 'phbp'
    rm = StaticRamanPhononsCalculator(Cbulk, BondPolarizability,
                                      calc=EMT(), name=name,
                                      delta=0.05, supercell=(2, 1, 1))
    rm.run()

    pz = PlaczekStaticPhonons(Cbulk, name=name, supercell=(2, 1, 1))
    energies_1kpt = pz.get_energies()
    pz.kpts = (2, 1, 1)
    energies_2kpts = pz.get_energies()
    assert len(energies_2kpts) == 2 * len(energies_1kpt)

    pz.kpts = (2, 1, 2)
    pz.summary()


def test_c3(testdir):
    """Can we calculate triangular (EMT groundstate) C3?"""
    y, z = 0.30646191, 1.14411339  # emt relaxed
    atoms = Atoms('C3', positions=[[0, 0, 0], [0, y, z], [0, z, y]])
    atoms.calc = EMT()

    name = 'bp'
    rm = StaticRamanCalculator(atoms, BondPolarizability,
                               name=name, exname=name, txt='-')
    rm.run()

    pz = PlaczekStatic(atoms, name=name)
    i_vib = pz.get_absolute_intensities()
    assert i_vib[-3:] == approx([5.36301901, 5.36680555, 35.7323934], 1e-6)
    pz.summary()
