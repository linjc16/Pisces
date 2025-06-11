import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ase.cell import Cell
from ase.geometry import minkowski_reduce

TOL = 1E-14


def test_issue():
    x = [[8.972058879514716, 0.0009788104586639142, 0.0005932485724084841],
         [4.485181755775297, 7.770520334862034, 0.00043663339838788054],
         [4.484671994095723, 2.5902066679984634, 16.25695615743613]]
    cell = Cell(x)
    cell.minkowski_reduce()


def test_cycle():
    # Without cycle-checking in the MR code, this cell causes failure
    a, b, c = 4.374006080444519, 2.0714140579127145, 3.671070851026261
    cell = np.array([[-a, b, c], [a, c, b], [a, b, c]])
    minkowski_reduce(cell)


@pytest.mark.parametrize("it", range(10))
def test_random(it):
    rng = np.random.RandomState(seed=it)
    B = rng.uniform(-1, 1, (3, 3))
    R, H = minkowski_reduce(B)
    assert_allclose(H @ B, R, atol=TOL)
    assert np.sign(np.linalg.det(B)) == np.sign(np.linalg.det(R))

    norms = np.linalg.norm(R, axis=1)
    assert (np.argsort(norms) == range(3)).all()

    # Test idempotency
    _, _H = minkowski_reduce(R)
    assert (_H == np.eye(3).astype(int)).all()

    rcell, _ = Cell(B).minkowski_reduce()
    assert_allclose(rcell, R, atol=TOL)


class TestKnownUnimodularMatrix():

    def setup_method(self):
        cell = np.array([[1, 1, 2], [0, 1, 4], [0, 0, 1]])
        unimodular = np.array([[1, 2, 2], [0, 1, 2], [0, 0, 1]])
        assert_almost_equal(np.linalg.det(unimodular), 1)
        self.lcell = unimodular.T @ cell

    @pytest.mark.parametrize("pbc", [1, True, (1, 1, 1)])
    def test_pbc(self, pbc):
        lcell = self.lcell
        rcell, _op = minkowski_reduce(lcell, pbc=pbc)
        assert_almost_equal(np.linalg.det(rcell), 1)

        rdet = np.linalg.det(rcell)
        ldet = np.linalg.det(lcell)
        assert np.sign(ldet) == np.sign(rdet)

    def test_0d(self):
        lcell = self.lcell
        rcell, _op = minkowski_reduce(lcell, pbc=[0, 0, 0])
        assert (rcell == lcell).all()    # 0D reduction does nothing

    @pytest.mark.parametrize("axis", range(3))
    def test_1d(self, axis):
        lcell = self.lcell
        rcell, _op = minkowski_reduce(lcell, pbc=np.roll([1, 0, 0], axis))
        assert (rcell == lcell).all()    # 1D reduction does nothing

        zcell = np.zeros((3, 3))
        zcell[0] = lcell[0]
        rcell, _ = Cell(zcell).minkowski_reduce()
        assert_allclose(rcell, zcell, atol=TOL)

    @pytest.mark.parametrize("axis", range(3))
    def test_2d(self, axis):
        lcell = self.lcell
        pbc = np.roll([0, 1, 1], axis)
        rcell, _op = minkowski_reduce(lcell.astype(float), pbc=pbc)
        assert (rcell[axis] == lcell[axis]).all()

        zcell = np.copy(lcell)
        zcell[axis] = 0
        rzcell, _ = Cell(zcell).minkowski_reduce()
        rcell[axis] = 0
        assert_allclose(rzcell, rcell, atol=TOL)

    def test_3d(self):
        lcell = self.lcell
        rcell, _op = minkowski_reduce(lcell)
        assert_almost_equal(np.linalg.det(rcell), 1)
