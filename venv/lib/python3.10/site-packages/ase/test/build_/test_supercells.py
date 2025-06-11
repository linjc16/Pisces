import itertools

import numpy as np
import pytest

from ase.build import bulk
from ase.build.supercells import (
    find_optimal_cell_shape,
    get_deviation_from_optimal_cell_shape,
    make_supercell,
)


@pytest.fixture()
def rng():
    return np.random.RandomState(seed=42)


@pytest.fixture(
    params=[
        bulk("NaCl", crystalstructure="rocksalt", a=4.0),
        bulk("NaCl", crystalstructure="rocksalt", a=4.0, cubic=True),
        bulk("Au", crystalstructure="fcc", a=4.0),
    ]
)
def prim(request):
    return request.param


@pytest.fixture(
    params=[
        3 * np.diag([1, 1, 1]),
        4 * np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        3 * np.diag([1, 2, 1]),
    ]
)
def P(request):
    return request.param


@pytest.fixture(params=["cell-major", "atom-major"])
def order(request):
    return request.param


def test_make_supercell(prim, P, order):
    n = int(round(np.linalg.det(P)))
    expected = n * len(prim)
    sc = make_supercell(prim, P, order=order)
    assert len(sc) == expected
    if order == "cell-major":
        symbols_expected = list(prim.symbols) * n
    elif order == "atom-major":
        symbols_expected = [s for s in prim.symbols for _ in range(n)]
    assert list(sc.symbols) == symbols_expected


def test_make_supercells_arrays(prim, P, order, rng):
    reps = int(round(np.linalg.det(P)))
    tags = list(range(len(prim)))
    momenta = rng.random((len(prim), 3))

    prim.set_tags(tags)
    prim.set_momenta(momenta)

    sc = make_supercell(prim, P, order=order)

    assert reps * len(prim) == len(sc.get_tags())
    if order == "cell-major":
        assert all(sc.get_tags() == np.tile(tags, reps))
        assert np.allclose(sc[: len(prim)].get_momenta(), prim.get_momenta())
        assert np.allclose(sc.get_momenta(), np.tile(momenta, (reps, 1)))
    elif order == "atom-major":
        assert all(sc.get_tags() == np.repeat(tags, reps))
        assert np.allclose(sc[::reps].get_momenta(), prim.get_momenta())
        assert np.allclose(sc.get_momenta(), np.repeat(momenta, reps, axis=0))


@pytest.mark.parametrize(
    "rep",
    [
        (1, 1, 1),
        (1, 2, 1),
        (4, 5, 6),
        (40, 19, 42),
    ],
)
def test_make_supercell_vs_repeat(prim, rep):
    P = np.diag(rep)

    at1 = prim * rep
    at1.wrap()
    at2 = make_supercell(prim, P, wrap=True)

    assert np.allclose(at1.positions, at2.positions)
    assert all(at1.symbols == at2.symbols)

    at1 = prim * rep
    at2 = make_supercell(prim, P, wrap=False)
    assert np.allclose(at1.positions, at2.positions)
    assert all(at1.symbols == at2.symbols)


@pytest.mark.parametrize(
    'cell, target_shape', (
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'sc'),
        ([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 'fcc'),
    )
)
def test_cell_metric_ideal(target_shape, cell):
    """Test cell with the ideal shape.

    Test if `get_deviation_from_optimal_cell_shape` returns perfect scores
    (0.0) for the ideal cells.
    Test also cell vectors with permutatation and elongation.
    """
    cell = np.asarray(cell)
    indices_permuted = itertools.permutations(range(3))
    elongations = range(1, 4)
    for perm, factor in itertools.product(indices_permuted, elongations):
        permuted_cell = [cell[i] * factor for i in perm]
        cell_metric = get_deviation_from_optimal_cell_shape(
            permuted_cell,
            target_shape=target_shape,
        )
        assert np.isclose(cell_metric, 0.0)


@pytest.mark.parametrize(
    'cell, target_shape', (
        ([[1, 0, 0], [0, 1, 0], [0, 0, 2]], 'sc'),
        ([[0, 1, 1], [1, 0, 1], [2, 2, 0]], 'fcc'),
    )
)
def test_cell_metric_twice_larger_lattice_vector(cell, target_shape):
    """Test cell with a twice larger lattice vector than the others.

    Test if `get_deviation_from_optimal_cell_shape` gives a correct value for
    the cells that have a lattice vector twice longer than the others.
    """
    # sqrt((1 - cbrt(2))**2 + (1 - cbrt(2))**2 + (2 - cbrt(2))**2) / cbrt(2)
    cell_metric_ref = 0.6558650332
    cell_metric = get_deviation_from_optimal_cell_shape(cell, target_shape)
    assert np.isclose(cell_metric, cell_metric_ref)


@pytest.mark.parametrize('target_shape', ['sc', 'fcc'])
def test_multiple_cells(target_shape: str) -> None:
    """Test if multiple cells can be evaluated at one time."""
    func = get_deviation_from_optimal_cell_shape
    cells = np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
        [[0, 1, 1], [1, 0, 1], [2, 2, 0]],
    ])
    metrics_separate = []
    for i in range(cells.shape[0]):
        metric = func(cells[i], target_shape)
        metrics_separate.append(metric)
    metrics_together = func(cells, target_shape)
    np.testing.assert_allclose(metrics_separate, metrics_together)


@pytest.mark.parametrize(
    'cell, target_shape', (
        ([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], 'sc'),
        ([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]], 'fcc'),
    )
)
def test_cell_metric_negative_determinant(cell, target_shape):
    """Test cell with negative determinant.

    Test if `get_deviation_from_optimal_cell_shape` works for the cells with
    negative determinants.
    """
    cell_metric = get_deviation_from_optimal_cell_shape(cell, target_shape)
    assert np.isclose(cell_metric, 0.0)


@pytest.mark.parametrize('cell, target_shape, target_size, ref_lengths', [
    (np.diag([1.0, 2.0, 4.0]), 'sc', 8, 4.0),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'fcc', 2, np.sqrt(2.0)),
    ([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 'sc', 4, 2.0),
])
def test_find_optimal_cell_shape(
        cell, target_shape, target_size, ref_lengths):
    """Test `find_optimal_cell_shape`.

    We test from sc to sc; from sc to fcc; and from fcc to sc."""
    supercell_matrix = find_optimal_cell_shape(cell, target_size, target_shape,
                                               lower_limit=-1, upper_limit=1)
    cell_metric = get_deviation_from_optimal_cell_shape(
        supercell_matrix @ cell,
        target_shape,
    )
    assert np.isclose(cell_metric, 0.0)
    cell_lengths = np.linalg.norm(np.dot(supercell_matrix, cell), axis=1)
    assert np.allclose(cell_lengths, ref_lengths)


def test_ideal_orientation() -> None:
    """Test if the ideal orientation is selected among candidates."""
    cell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    target_size = 2
    target_shape = 'fcc'
    supercell_matrix_ref = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    supercell_matrix = find_optimal_cell_shape(cell, target_size, target_shape)
    np.testing.assert_array_equal(supercell_matrix, supercell_matrix_ref)

    cell = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    target_size = 4
    target_shape = 'sc'
    supercell_matrix_ref = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    supercell_matrix = find_optimal_cell_shape(cell, target_size, target_shape)
    np.testing.assert_array_equal(supercell_matrix, supercell_matrix_ref)
