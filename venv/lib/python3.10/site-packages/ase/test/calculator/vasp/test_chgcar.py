"""Tests for CHG/CHGCAR."""
import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.calculators.vasp import VaspChargeDensity


def test_chgcar(datadir):
    """Test if a CHG/CHGCAR file can be parsed correctly."""
    charge_density = VaspChargeDensity(datadir / 'vasp/Li/CHG')
    atoms = charge_density.atoms[0]
    chg = charge_density.chg[0]
    assert not compare_atoms(atoms, bulk('Li', a=3.49))
    assert np.mean(chg) * atoms.get_volume() == pytest.approx(1.0, rel=1e-3)
