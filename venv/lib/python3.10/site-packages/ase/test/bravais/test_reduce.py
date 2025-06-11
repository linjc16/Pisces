import numpy as np
import pytest

from ase import Atoms
from ase.build.tools import reduce_lattice
from ase.calculators.emt import EMT


def test_reduce_lattice():
    """Test that reduce_lattice() correctly in-place-reduces lattice.

    We test that the 60 degree hex cell reduces to 120 degree cell.
    To verify that the full system is physically equivalent, we verify
    that the EMT energy is the same after the transformation."""

    origcellpar = [3, 3, 4, 90, 90, 60]
    ref_cellpar = list(origcellpar)
    ref_cellpar[-1] = 120

    symbols = 'CuAgAuPt'
    scaled_positions = np.random.RandomState(42).random((4, 3))

    atoms = Atoms(symbols=symbols,
                  scaled_positions=scaled_positions,
                  cell=origcellpar, pbc=True)

    reduced_atoms = atoms.copy()
    reduce_lattice(reduced_atoms)

    assert reduced_atoms.cell.cellpar() == pytest.approx(ref_cellpar)
    assert emt_energy(atoms) == pytest.approx(emt_energy(reduced_atoms))


def emt_energy(atoms):
    atoms = atoms.copy()
    atoms.calc = EMT()
    return atoms.get_potential_energy()
