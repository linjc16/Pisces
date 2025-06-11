import itertools

import numpy as np
import pytest

from ase.cell import Cell


def test_niggli_0d():
    rcell, op = Cell.new().niggli_reduce()
    assert rcell.rank == 0
    assert (op == np.identity(3, dtype=int)).all()


def test_niggli_1d():
    cell = Cell.new()
    vector = [1, 2, 3]
    cell[1] = vector

    rcell, op = cell.niggli_reduce()
    assert rcell.lengths() == pytest.approx([np.linalg.norm(vector), 0, 0])
    assert Cell(op.T @ cell).cellpar() == pytest.approx(rcell.cellpar())


def test_niggli_2d():
    cell = Cell.new()
    cell[0] = [3, 4, 5]
    cell[2] = [5, 6, 7]
    rcell, op = cell.niggli_reduce()
    assert rcell.rank == 2
    assert rcell.lengths()[2] == 0
    assert Cell(op.T @ cell).cellpar() == pytest.approx(rcell.cellpar())


@pytest.mark.parametrize('npbc', [0, 1, 2, 3])
@pytest.mark.parametrize('perm', itertools.permutations(range(3)))
def test_niggli_atoms_ndim(npbc, perm):
    from ase.build import fcc111, niggli_reduce
    from ase.calculators.emt import EMT
    from ase.geometry.geometry import permute_axes

    perm = np.array(perm)

    atoms = fcc111('Au', (2, 3, 1), vacuum=2.0)
    atoms.pbc = False
    atoms.pbc[:npbc] = True
    atoms.cell[2] = [0, 0, 1]
    if npbc == 1:
        atoms.cell[1] = [0, 0, 0]
    atoms.rattle(stdev=0.1)
    atoms = permute_axes(atoms, perm)
    atoms.calc = EMT()
    e1 = atoms.get_potential_energy()
    niggli_reduce(atoms)
    e2 = atoms.get_potential_energy()

    assert e2 == pytest.approx(e1, abs=1e-10)


def test_no_nonorthogonal_niggli():
    from ase.build import bulk, niggli_reduce
    atoms = bulk('Au')
    atoms.pbc[1] = False
    with pytest.raises(ValueError, match='Non-orthogonal'):
        niggli_reduce(atoms)
