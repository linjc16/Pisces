import numpy as np
import pytest

from ase.calculators.h2morse import (
    De,
    Etrans,
    H2Morse,
    H2MorseCalculator,
    H2MorseExcitedStates,
    H2MorseExcitedStatesAndCalculator,
    H2MorseExcitedStatesCalculator,
    Re,
    ome,
)
from ase.vibrations import Vibrations


def test_gs_minimum(testdir):
    """Test ground state minimum distance, energy and
    vibrational frequency"""
    atoms = H2Morse()
    assert atoms.get_distance(0, 1) == pytest.approx(Re[0], 1.e-12)
    assert atoms.get_potential_energy() == -De[0]
    # check ground state vibrations
    vib = Vibrations(atoms)
    vib.run()
    assert (vib.get_frequencies().real[-1] ==
            pytest.approx(ome[0], 1e-2))


def test_gs_io_overlap(testdir):
    """Test ground state IO and 'wave function' overlap"""
    atoms0 = H2Morse()
    calc0 = atoms0.calc
    fname = 'calc0'
    calc0.write(fname)
    calc1 = H2MorseCalculator(fname)
    for wf0, wf1 in zip(calc0.wfs, calc1.wfs):
        assert wf0 == pytest.approx(wf1, 1e-5)

    atoms1 = H2Morse()
    ov = calc0.overlap(calc0)
    # own overlap is the unity matrix
    assert np.eye(4) == pytest.approx(calc0.overlap(calc0), 1e-8)
    # self and other - test on unitarity
    ov = calc0.overlap(atoms1.calc)
    assert np.eye(4) == pytest.approx(ov.dot(ov.T), 1e-8)


def test_excited_state():
    """Test excited state transition energies"""
    gsatoms = H2Morse()
    Egs0 = gsatoms.get_potential_energy()
    for i in range(1, 4):
        exatoms = H2Morse()
        exatoms[1].position[2] = Re[i]  # set to potential minimum
        Egs = exatoms.get_potential_energy()

        exc = H2MorseExcitedStatesCalculator()
        exl = exc.calculate(exatoms)
        assert (exl[i - 1].energy ==
                pytest.approx(Etrans[i] - Egs + Egs0, 1e-8))


def test_excited_io(testdir):
    """Check writing and reading"""
    fname = 'exlist.dat'
    atoms = H2Morse()
    exc = H2MorseExcitedStatesCalculator()
    exl1 = exc.calculate(atoms)
    exl1.write(fname)

    exl2 = H2MorseExcitedStates(fname)
    for ex1, ex2 in zip(exl1, exl2):
        assert ex1.energy == pytest.approx(ex2.energy, 1e-3)
        assert ex1.mur == pytest.approx(ex2.mur, 1e-5)
        assert ex1.muv == pytest.approx(ex2.muv, 1e-5)


def test_traditional(testdir):
    """Check that traditional calling works"""
    atoms = H2Morse()
    fname = 'exlist.dat'
    exl1 = H2MorseExcitedStatesAndCalculator(atoms.calc)
    exl1.write(fname)
    ex1 = exl1[0]

    exl2 = H2MorseExcitedStatesAndCalculator(fname, nstates=1)
    ex2 = exl2[-1]
    assert ex1.energy == pytest.approx(ex2.energy, 1e-3)
    assert ex1.mur == pytest.approx(ex2.mur, 1e-5)
    assert ex1.muv == pytest.approx(ex2.muv, 1e-5)
