import numpy as np

from ase import Atoms
from ase.geometry import get_duplicate_atoms


def test_atoms_get_duplicates():

    at = Atoms('H5', positions=[[0., 0., 0.],
                                [1., 0., 0.],
                                [1.01, 0, 0],
                                [3, 2.2, 5.2],
                                [0.1, -0.01, 0.1]])

    dups = get_duplicate_atoms(at)
    assert all((dups == [[1, 2]]).tolist()) is True

    dups = get_duplicate_atoms(at, cutoff=0.2)
    assert all((dups == [[0, 4], [1, 2]]).tolist()) is True

    get_duplicate_atoms(at, delete=True)
    assert len(at) == 4


def test_no_duplicate_atoms():
    """test if it works if no duplicates are detected."""
    at = Atoms('H3', positions=[[0., 0., 0.],
                                [1., 0., 0.],
                                [3, 2.2, 5.2]])

    get_duplicate_atoms(at, delete=True)
    dups = get_duplicate_atoms(at)

    assert dups.size == 0


def test_pbc():
    """test if it works under PBCs."""
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
    atoms = Atoms('H2', positions=positions, cell=np.eye(3), pbc=True)
    dups = get_duplicate_atoms(atoms, cutoff=0.2)
    np.testing.assert_array_equal(dups, [[0, 1]])
