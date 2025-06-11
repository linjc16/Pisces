import numpy as np
from numpy.testing import assert_allclose

from ase.cell import Cell
from ase.lattice import all_variants


def test_standard_form():
    TOL = 1E-10
    for lat in all_variants():
        cell0 = lat.tocell()
        for sign in [-1, 1]:
            cell = Cell(sign * cell0)
            # lower triangular form
            rcell, Q = cell.standard_form(form='lower')
            assert_allclose(rcell @ Q, cell, atol=TOL)
            assert_allclose(np.linalg.det(rcell), np.linalg.det(cell))
            assert_allclose(rcell.ravel()[[1, 2, 5]], 0, atol=TOL)
            # upper triangular form
            rcell, Q = cell.standard_form(form='upper')
            assert_allclose(rcell @ Q, cell, atol=TOL)
            assert_allclose(np.linalg.det(rcell), np.linalg.det(cell))
            assert_allclose(rcell.ravel()[[3, 6, 7]], 0, atol=TOL)
