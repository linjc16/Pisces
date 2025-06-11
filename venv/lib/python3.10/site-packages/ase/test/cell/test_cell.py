import numpy as np
import pytest

from ase.cell import Cell

testcellpar = (2, 3, 4, 50, 60, 70)


@pytest.fixture()
def cell():
    return Cell.new(testcellpar)


def test_lengths_angles(cell):
    assert cell.cellpar() == pytest.approx(testcellpar)
    assert cell.lengths() == pytest.approx(testcellpar[:3])
    assert cell.angles() == pytest.approx(testcellpar[3:])


def test_new():
    assert np.array_equal(Cell.new(), np.zeros((3, 3)))
    assert np.array_equal(Cell.new([1, 2, 3]), np.diag([1, 2, 3]))
    assert Cell.new(testcellpar).cellpar() == pytest.approx(testcellpar)
    arr = np.arange(9).reshape(3, 3)
    assert np.array_equal(Cell.new(arr), arr)
    with pytest.raises(ValueError):
        Cell.new([1, 2, 3, 4])


def test_handedness(cell):
    assert cell.handedness == 1
    cell[0] *= -1
    assert cell.handedness == -1
    cell[0] = 0
    assert cell.handedness == 0


@pytest.fixture()
def randcell():
    rng = np.random.RandomState(42)
    return Cell(rng.random((3, 3)))


def test_normal(randcell):
    normals = randcell.normals()

    for i in range(3):
        normal = randcell.normal(i)
        assert normal == pytest.approx(normals[i])

        for j in range(3):
            if i == j:
                ref = np.cross(randcell[j - 2], randcell[j - 1])
                assert normal == pytest.approx(ref)
            else:
                assert abs(randcell[j] @ normal) < 1e-14


def test_area(randcell):
    areas = randcell.areas()
    for i in range(3):
        area = randcell.area(i)
        assert area == pytest.approx(areas[i])
        randcell[i - 2] @ randcell[i - 1]


@pytest.mark.parametrize(
    'zeromask',
    [[], [1], [0, 2], [0, 1, 2]],
    ids=lambda mask: f'dim={3 - len(mask)}'
)
def test_reciprocal_ndim(randcell, zeromask):
    randcell[zeromask] = 0
    ndim = 3 - len(zeromask)
    assert randcell.rank == ndim
    reciprocal = randcell.reciprocal()
    assert reciprocal.rank == ndim

    ref = np.identity(3)
    ref[zeromask] = 0
    assert reciprocal @ randcell.T == pytest.approx(ref)


def test_total_area(randcell):
    lengths = randcell.lengths()
    sin_angles = np.sin(np.radians(randcell.angles()))
    areas = randcell.areas()

    for i in range(3):
        area = lengths[i - 2] * lengths[i - 1] * sin_angles[i]
        assert area == pytest.approx(areas[i])


def test_cell_edit_via_view():
    cell = Cell(np.arange(9).reshape(3, 3))

    # np.reshape() is a no-op so it should not copy by default:
    arr = np.reshape(cell, (3, 3))
    arr[-1] = 42
    assert cell[-1, -1] == 42

    # np.array() should copy, so edit will not be inplace:
    cell1 = np.array(cell)
    cell1[-1, -1] = 64
    assert cell[-1, -1] == 42

    # This should be in-place:
    cell1 = np.array(cell, copy=False)
    cell[-1, -1] = 64
    assert cell[-1, -1] == 64
