import numpy as np
import pytest
from numpy.testing import assert_allclose

from ase.geometry import find_mic
from ase.lattice.cubic import FaceCenteredCubic
from ase.neighborlist import NeighborList, PrimitiveNeighborList
from ase.neighborlist import mic as NeighborListMic


@pytest.mark.parametrize("dim", [2, 3])
def test_minimum_image_convention(dim):
    size = 2
    if dim == 2:
        pbc = [True, True, False]
    else:
        pbc = [True, True, True]
    atoms = FaceCenteredCubic(size=[size, size, size],
                              symbol='Cu',
                              latticeconstant=2,
                              pbc=pbc)
    if dim == 2:
        cell = atoms.cell.uncomplete(atoms.pbc)
        atoms.set_cell(cell, scale_atoms=False)

    d0 = atoms.get_distances(0, np.arange(len(atoms)), mic=True)

    if dim == 2:
        U = np.array([[1, 2, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    else:
        U = np.array([[1, 2, 2],
                      [0, 1, 2],
                      [0, 0, 1]])
    assert np.linalg.det(U) == 1
    atoms.set_cell(U.T @ atoms.cell, scale_atoms=False)
    atoms.wrap()

    d1 = atoms.get_distances(0, np.arange(len(atoms)), mic=True)
    assert_allclose(d0, d1)

    vnbrlist = NeighborListMic(atoms.get_positions(), atoms.cell, atoms.pbc)
    d2 = np.linalg.norm(vnbrlist, axis=1)
    assert_allclose(d0, d2)

    nl = NeighborList(np.ones(len(atoms)) * 2 * size * np.sqrt(3),
                      bothways=True,
                      primitive=PrimitiveNeighborList)
    nl.update(atoms)
    indices, offsets = nl.get_neighbors(0)

    d3 = float("inf") * np.ones(len(atoms))
    for i, offset in zip(indices, offsets):
        p = atoms.positions[i] + offset @ atoms.get_cell()
        d = np.linalg.norm(p - atoms.positions[0])
        d3[i] = min(d3[i], d)
    assert_allclose(d0, d3)


def test_numpy_array():
    # Tests Issue #787
    atoms = FaceCenteredCubic(size=[1, 1, 1],
                              symbol='Cu',
                              latticeconstant=2,
                              pbc=True)

    find_mic(atoms.positions, np.array(atoms.cell), pbc=True)
