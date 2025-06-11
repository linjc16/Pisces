import numpy as np
import pytest

from ase.cell import Cell


@pytest.fixture()
def cell():
    return Cell([[1., 0., 0.],
                 [.1, 1., 0.],
                 [0., 0., 0.]])


def test_obl(cell):
    """Verify 2D Bravais lattice and band path versus pbc information."""
    lat = cell.get_bravais_lattice()
    print(cell.cellpar())
    print(lat)
    assert lat.name == 'OBL'


def test_mcl_obl(cell):
    cell[2, 2] = 7
    lat3d = cell.get_bravais_lattice()
    print(lat3d)
    assert lat3d.name == 'MCL'
    lat2d_pbc = cell.get_bravais_lattice(pbc=[1, 1, 0])
    print(lat2d_pbc)
    assert lat2d_pbc.name == 'OBL'

    path = cell.bandpath()
    print(path)

    path2d = cell.bandpath(pbc=[1, 1, 0])
    print(path2d)
    assert path2d.cell.rank == 2
    assert path2d.cell.get_bravais_lattice().name == 'OBL'


@pytest.mark.parametrize('angle', [60, 120])
def test_2d_bandpath_handedness(angle):
    """Test that the x/y part is right-handed in 2D lattice.

    During lattice determination, the whole 3x3 matrix is right-handed
    including "dummy" z axis in 2D cells.  However this did not
    guarantee that the x/y part itself was right-handed, and this
    test would fail for angle=60."""

    cell = Cell.new([1, 1, 0, 90, 90, angle])
    assert cell.get_bravais_lattice().name == 'HEX2D'
    bandpath = cell.bandpath()

    assert bandpath.cell.rank == 2
    assert np.linalg.det(bandpath.cell[:2, :2]) > 0


def test_2d_handedness_obl():
    # Previous code had a bug where left-handed cell would pick the lattice
    # object corresponding to the wrong back-transformation.
    dalpha = 5.1234
    cell = Cell.new([2, 1, 0, 90, 90, 90 + dalpha])
    lat = cell.get_bravais_lattice()
    assert lat.alpha == pytest.approx(90 - dalpha)
