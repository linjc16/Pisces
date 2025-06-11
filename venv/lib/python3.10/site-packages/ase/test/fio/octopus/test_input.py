import numpy as np
import pytest

from ase.build import bulk, fcc111, molecule
from ase.io.octopus.input import atoms2kwargs
from ase.units import Bohr


def getcoords(block):
    words = [line[1:] for line in block]
    return np.array(words).astype(float)


def test_molecule():
    atoms = molecule('H2O')
    kwargs = atoms2kwargs(atoms, use_ase_cell=False)
    assert atoms.positions == pytest.approx(
        getcoords(kwargs['coordinates']) * Bohr)
    # assert 'boxshape' not in kwargs and 'latticevectors' not in kwargs


def test_molecule_box():
    atoms = molecule('H2O', vacuum=3.0)
    kwargs = atoms2kwargs(atoms, use_ase_cell=True)
    lsize = np.array(kwargs['lsize'], float)[0]
    # assert kwargs['boxshape'] == 'parallelepiped'
    # note: lsize are the "half lengths", box is [-lsize, lsize]:
    assert atoms.cell.lengths() == pytest.approx(2 * lsize * Bohr)

    coords = getcoords(kwargs['coordinates']) * Bohr
    cell_center = 0.5 * atoms.cell.sum(axis=0)
    assert atoms.positions - cell_center == pytest.approx(coords)


def compare_scaled(atoms, kwargs):
    assert np.array(kwargs['latticevectors'], float) == pytest.approx(
        atoms.cell / Bohr)
    assert getcoords(kwargs['reducedcoordinates']) == pytest.approx(
        atoms.get_scaled_positions())


def test_2d_surface():
    atoms = fcc111('Au', size=(1, 1, 1), vacuum=4.0)
    kwargs = atoms2kwargs(atoms, use_ase_cell=True)
    compare_scaled(atoms, kwargs)
    assert kwargs['periodicdimensions'] == 2


def test_3d_periodic():
    atoms = bulk('Ti')
    kwargs = atoms2kwargs(atoms, use_ase_cell=True)
    compare_scaled(atoms, kwargs)
    assert kwargs['periodicdimensions'] == 3
