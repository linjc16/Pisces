import numpy as np
import pytest
from numpy.testing import assert_allclose

import ase.lattice
from ase.spacegroup import (
    Spacegroup,
    get_bravais_class,
    get_point_group,
    polar_space_group,
)
from ase.spacegroup.spacegroup import SpacegroupNotFoundError

TOL = 1E-10

functions = [get_bravais_class, get_point_group, polar_space_group]


@pytest.mark.parametrize("sg,lattice,point_group,polar",
                         [[100, ase.lattice.TET, '4mm', True],
                          [225, ase.lattice.FCC, '4/m -3 2/m', False]])
def test_valid_spacegroup(sg, lattice, point_group, polar):
    assert get_bravais_class(sg) == lattice
    assert get_point_group(sg) == point_group
    assert polar_space_group(sg) == polar


@pytest.mark.parametrize("func", functions)
def test_nonpositive_spacegroup(func):
    with pytest.raises(ValueError, match="positive"):
        func(0)


@pytest.mark.parametrize("func", functions)
def test_bad_spacegroup(func):
    with pytest.raises(ValueError, match="Bad"):
        func(400)


def _spacegroup_reciprocal_cell(no, setting):
    try:
        sg = Spacegroup(no, setting)
    except SpacegroupNotFoundError:
        return
    reciprocal_check = np.linalg.inv(sg.scaled_primitive_cell).T
    assert_allclose(sg.reciprocal_cell, reciprocal_check, atol=TOL)


def test_spacegroup_reciprocal_cell():
    # Would be better to use pytest.mark.parametrize if we can figure out
    # how to not list hundreds of test cases in the output.
    for setting in [1, 2]:
        for no in range(1, 231):
            _spacegroup_reciprocal_cell(setting, no)
