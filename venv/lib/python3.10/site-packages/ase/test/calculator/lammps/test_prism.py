"""Test Prism"""
from math import sqrt

import numpy as np
import pytest

from ase import Atoms
from ase.calculators.lammps import Prism
from ase.calculators.lammps.coordinatetransform import (
    calc_box_parameters,
    calc_reduced_cell,
)


def make_array(structure: str) -> np.ndarray:
    """Make array for the given structure"""
    if structure == "sc":
        array = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    elif structure == "bcc":
        array = np.array([
            [-0.5, +0.5, +0.5],
            [+0.5, -0.5, +0.5],
            [+0.5, +0.5, -0.5],
        ])
    elif structure == "fcc":
        array = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
    elif structure == "hcp":
        covera = sqrt(8.0 / 3.0)
        array = np.array([
            [0.5, -0.5 * sqrt(3.0), 0.0],
            [0.5, +0.5 * sqrt(3.0), 0.0],
            [0.0, 0.0, covera],
        ])
    else:
        raise ValueError(structure)
    return array


class TestCalcBoxParameters:
    """Test calc_box_parameters"""

    def test_sc(self):
        """Test sc"""
        array = make_array("sc")
        box = calc_box_parameters(array)
        box_ref = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(box, box_ref)

    def test_bcc(self):
        """Test bcc"""
        array = make_array("bcc")
        box = calc_box_parameters(array)
        box_ref = np.array([
            +0.8660254038,  # +sqrt(3) / 2
            +0.8164965809,  # +sqrt(6) / 3
            +0.7071067812,  # +sqrt(2) / 2
            -0.2886751346,  # -sqrt(3) / 6
            -0.2886751346,  # -sqrt(3) / 6
            -0.4082482905,  # -sqrt(6) / 6
        ])
        np.testing.assert_allclose(box, box_ref)

    def test_fcc(self):
        """Test fcc"""
        array = make_array("fcc")
        box = calc_box_parameters(array)
        box_ref = np.array([
            +0.7071067812,  # +sqrt(2) / 2
            +0.6123724357,  # +sqrt(6) / 4
            +0.5773502692,  # +sqrt(3) / 3
            +0.3535533906,  # +sqrt(2) / 4
            +0.3535533906,  # +sqrt(2) / 4
            +0.2041241452,  # +sqrt(6) / 12
        ])
        np.testing.assert_allclose(box, box_ref)

    def test_hcp(self):
        """Test hcp"""
        covera = sqrt(8.0 / 3.0)
        array = make_array("hcp")
        box = calc_box_parameters(array)
        box_ref = np.array([1.0, 0.5 * sqrt(3.0), covera, -0.5, 0.0, 0.0])
        np.testing.assert_allclose(box, box_ref)


class TestReducedCell:
    """Test reduced cell"""

    def test_small(self):
        """Test small tilt"""
        original = ((3.0, 0.0, 0.0), (-1.0, 3.0, 0.0), (0.0, 0.0, 3.0))
        reduced_ref = ((3.0, 0.0, 0.0), (-1.0, 3.0, 0.0), (0.0, 0.0, 3.0))
        self.check(original, reduced_ref)

    def test_large(self):
        """Test large tilt"""
        original = ((3.0, 0.0, 0.0), (2.0, 3.0, 0.0), (0.0, 0.0, 3.0))
        reduced_ref = ((3.0, 0.0, 0.0), (-1.0, 3.0, 0.0), (0.0, 0.0, 3.0))
        self.check(original, reduced_ref)

    def test_very_large(self):
        """Test very large tilt"""
        original = ((3.0, 0.0, 0.0), (5.0, 3.0, 0.0), (0.0, 0.0, 3.0))
        reduced_ref = ((3.0, 0.0, 0.0), (-1.0, 3.0, 0.0), (0.0, 0.0, 3.0))
        self.check(original, reduced_ref)

    def test_xz_and_yz_0(self):
        """Test if the update of yz also modifies xz correctly"""
        original = ((3.0, 0.0, 0.0), (1.0, 3.0, 0.0), (0.0, 2.0, 3.0))
        # The third vector should be updated as c -> c - b
        reduced_ref = ((3.0, 0.0, 0.0), (1.0, 3.0, 0.0), (-1.0, -1.0, 3.0))
        self.check(original, reduced_ref)

    def test_xz_and_yz_1(self):
        """Test if the update of yz also modifies xz correctly"""
        original = ((3.0, 0.0, 0.0), (1.0, 3.0, 0.0), (-1.0, 2.0, 3.0))
        # The third vector should be updated as c -> c - b -> c - b + a
        reduced_ref = ((3.0, 0.0, 0.0), (1.0, 3.0, 0.0), (1.0, -1.0, 3.0))
        self.check(original, reduced_ref)

    def check(self, original: np.ndarray, reduced_ref: np.ndarray):
        """Check"""
        reduced = calc_reduced_cell(np.array(original), pbc=(True, True, True))
        np.testing.assert_allclose(reduced, reduced_ref)


@pytest.mark.parametrize("wrap", (False, True))
@pytest.mark.parametrize("reduce_cell", (False, True))
@pytest.mark.parametrize("pbc", (False, True))
@pytest.mark.parametrize("structure", ("sc", "bcc", "fcc", "hcp"))
def test_vectors(structure: str, pbc: bool, reduce_cell: bool, wrap: bool):
    """Test if vector conversion works as expected"""
    array = make_array(structure)
    rng = np.random.default_rng(42)
    positions = 20.0 * rng.random((10, 3)) - 10.0
    atoms = Atoms(positions=positions, cell=array, pbc=pbc)
    prism = Prism(array, pbc=pbc, reduce_cell=reduce_cell)
    vectors_ref = atoms.get_positions(wrap=(wrap or reduce_cell))
    vectors = prism.vector_to_lammps(vectors_ref, wrap=wrap)
    vectors = prism.vector_to_ase(vectors, wrap=wrap)
    np.testing.assert_allclose(vectors, vectors_ref)
