"""Tests for the CP2K ASE calculator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

import pytest

from ase import units
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS


@pytest.fixture()
def atoms():
    return molecule('H2', vacuum=2.0)


def test_geoopt(cp2k_factory, atoms):
    calc = cp2k_factory.calc(label='test_H2_GOPT', print_level='LOW')
    atoms.calc = calc

    with BFGS(atoms, logfile=None) as gopt:
        gopt.run(fmax=1e-6)

    dist = atoms.get_distance(0, 1)
    dist_ref = 0.7245595
    assert (dist - dist_ref) / dist_ref < 1e-7

    energy_ref = -30.7025616943
    energy = atoms.get_potential_energy()
    assert (energy - energy_ref) / energy_ref < 1e-10


def test_h2_lda(cp2k_factory, atoms):
    calc = cp2k_factory.calc(label='test_H2_LDA')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -30.6989595886
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10


def test_h2_libxc(cp2k_factory, atoms):
    calc = cp2k_factory.calc(
        xc='XC_GGA_X_PBE XC_GGA_C_PBE',
        pseudo_potential="GTH-PBE",
        label='test_H2_libxc')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -31.591716529642
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10


def test_h2_ls(cp2k_factory, atoms):
    inp = """&FORCE_EVAL
               &DFT
                 &QS
                   LS_SCF ON
                 &END QS
               &END DFT
             &END FORCE_EVAL"""
    calc = cp2k_factory.calc(label='test_H2_LS', inp=inp)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -30.6989581747
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 5e-7


def test_h2_pbe(cp2k_factory, atoms):
    calc = cp2k_factory.calc(xc='PBE', label='test_H2_PBE')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -31.5917284949
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10


def test_md(cp2k_factory):
    calc = cp2k_factory.calc(label='test_H2_MD')
    positions = [(0, 0, 0), (0, 0, 0.7245595)]
    atoms = Atoms('HH', positions=positions, calculator=calc)
    atoms.center(vacuum=2.0)

    MaxwellBoltzmannDistribution(atoms, temperature_K=0.5 * 300,
                                 force_temp=True)
    energy_start = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    with VelocityVerlet(atoms, 0.5 * units.fs) as dyn:
        dyn.run(20)

    energy_end = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    assert abs(energy_start - energy_end) < 1e-4


def test_o2(cp2k_factory):
    calc = cp2k_factory.calc(
        label='test_O2', uks=True, cutoff=150 * units.Rydberg,
        basis_set="SZV-MOLOPT-SR-GTH", multiplicity=3)
    o2 = molecule('O2', calculator=calc)
    o2.center(vacuum=2.0)
    energy = o2.get_potential_energy()
    energy_ref = -862.8384369579051
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10


def test_restart(cp2k_factory, atoms):
    calc = cp2k_factory.calc()
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('test_restart')  # write a restart
    calc2 = cp2k_factory.calc(restart='test_restart')  # load a restart
    assert not calc2.calculation_required(atoms, ['energy'])


def test_unknown_keywords(cp2k_factory):
    with pytest.raises(CalculatorSetupError):
        cp2k_factory.calc(dummy_nonexistent_keyword='hello')


def test_close(cp2k_factory, atoms):
    """Ensure we cleanly close the calculator and then restart it"""

    # The calculator starts on
    calc = cp2k_factory.calc(label='test_H2_GOPT', print_level='LOW')
    assert calc._shell is not None
    calc.get_potential_energy(atoms)  # Make sure it runs

    # It is shut down by the call
    assert calc._shell is not None
    child = calc._shell._child
    calc.close()
    assert child.poll() == 0

    # Ensure it starts back up
    atoms.rattle(0.01)
    calc.get_potential_energy(atoms)
    assert calc._shell is not None
    calc.close()


def test_context(cp2k_factory, atoms):
    """Ensure we can use the CP2K shell as a context manager"""

    with cp2k_factory.calc(label='test_H2_GOPT', print_level='LOW') as calc:
        atoms.calc = calc
        atoms.get_potential_energy()
        child = calc._shell._child
    assert child.poll() == 0


@pytest.mark.xfail()  # Will pass with the 2024.2 version of CP2K
def test_set_pos_file(cp2k_factory, atoms):
    """Test passing coordinates via file rather than stdin

    This will pass when testing against a new version of CP2K.
    When that happens, remove the `xfail` decorator
    and change 2024.X in `cp2k.py` to the new version number. -wardlt
    """

    with cp2k_factory.calc(label='test_H2_GOPT', print_level='LOW',
                           set_pos_file=True) as calc:
        atoms.calc = calc
        atoms.get_potential_energy()
