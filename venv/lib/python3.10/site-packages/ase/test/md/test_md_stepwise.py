"""Tests for `MolecularDynmics.irun`."""

import pytest

from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.md.npt import NPT
from ase.units import fs


@pytest.fixture(name='atoms0')
def fixture_atoms0():
    """Make `atoms0`."""
    atoms = bulk('Au', cubic=True)
    atoms.rattle(stdev=0.15)
    return atoms


@pytest.fixture(name='atoms')
def fixture_atoms(atoms0):
    """Make `atoms`."""
    atoms = atoms0.copy()
    atoms.calc = EMT()
    return atoms


def test_irun_start(atoms0: Atoms, atoms: Atoms) -> None:
    """Test if `irun` works."""
    with NPT(atoms, timestep=1.0 * fs, temperature_K=1000.0) as md:
        irun = md.irun(steps=10)
        next(irun)  # Initially it yields without yet having performed a step:
        assert not compare_atoms(atoms0, atoms)
        next(irun)  # Now it must have performed a step:
        assert compare_atoms(atoms0, atoms) == ['positions']
