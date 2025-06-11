import pytest

from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked


def test_miller_lindep():
    with pytest.raises(ValueError):
        # The Miller indices of the surfaces are linearly dependent
        FaceCenteredCubic(symbol='Cu',
                          miller=[[1, 1, 0], [1, 1, 0], [0, 0, 1]])


def test_fcc_ok():
    atoms = FaceCenteredCubic(symbol='Cu',
                              miller=[[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    print(atoms.get_cell())


@pytest.mark.parametrize('directions', [
    [[1, 1, 0], [1, 1, 0], [0, 0, 1]],
    [[1, 1, 0], [1, 0, 0], [0, 1, 0]]
])
def test_fcc_directions_linearly_dependent(directions):
    # The directions spanning the unit cell are linearly dependent
    with pytest.raises(ValueError):
        FaceCenteredCubic(symbol='Cu', directions=directions)


def test_fcc_directions_ok():
    atoms = FaceCenteredCubic(symbol='Cu',
                              directions=[[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    print(atoms.get_cell())


def test_hcp_miller_lienarly_dependent():
    with pytest.raises((ValueError, NotImplementedError)):
        # The Miller indices of the surfaces are linearly dependent
        HexagonalClosedPacked(symbol='Mg',
                              miller=[[1, -1, 0, 0],
                                      [1, 0, -1, 0],
                                      [0, 1, -1, 0]])

    # This one should be OK
    #
    # It is not!  The miller argument is broken in hexagonal crystals!
    #
    # atoms = HexagonalClosedPacked(symbol='Mg',
    #                               miller=[[1, -1, 0, 0],
    #                                       [1, 0, -1, 0],
    #                                       [0, 0, 0, 1]])
    # print(atoms.get_cell())


def test_hcp_cell_linearly_dependent():
    with pytest.raises(ValueError):
        # The directions spanning the unit cell are linearly dependent
        HexagonalClosedPacked(symbol='Mg',
                              directions=[[1, -1, 0, 0],
                                          [1, 0, -1, 0],
                                          [0, 1, -1, 0]])


def test_hcp():
    atoms = HexagonalClosedPacked(symbol='Mg',
                                  directions=[[1, -1, 0, 0],
                                              [1, 0, -1, 0],
                                              [0, 0, 0, 1]])
    print(atoms.get_cell())
