import numpy as np

from ase import Atoms

pbc = [1, 1, 0]
cell = [[1, 0, 0], [0, 1, 0], [0, 0, 4]]

positions = [[-0.1, 1.01, -0.5]]
positions_wrapped = [[0.9, 0.01, -0.5]]

atoms = Atoms("H", positions=positions, cell=cell, pbc=pbc)


def test_positions(atoms=atoms):
    assert np.allclose(positions, atoms.get_positions())


def test_positions_wrapped(atoms=atoms):
    assert np.allclose(positions_wrapped, atoms.get_positions(wrap=True))


def test_wrapped_positions(atoms=atoms):
    atoms.wrap()
    assert np.allclose(positions_wrapped, atoms.get_positions())
