import warnings

import numpy as np
import pytest
from pytest import approx

from ase.atoms import Atoms
from ase.build import bulk
from ase.constraints import (
    FixAtoms,
    FixCartesian,
    FixCartesianParametricRelations,
    FixScaledParametricRelations,
)
from ase.io.aims import parse_geometry_lines
from ase.io.aims import read_aims as read

format = "aims"

file = "geometry.in"


@pytest.fixture()
def Si():
    return bulk("Si")


@pytest.fixture()
def H2O():
    return Atoms("H2O", [(0.9584, 0.0, 0.0),
                 (-0.2400, 0.9279, 0.0), (0.0, 0.0, 0.0)])


def test_cartesian_Si(Si):
    """write cartesian coords and check if structure was preserved"""
    Si.write(file, format=format)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)


def test_scaled_Si(Si):
    """write fractional coords and check if structure was preserved"""
    Si.write(file, format=format, scaled=True, wrap=False)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)


def test_param_const_Si(Si):
    """Check to ensure parametric constraints are passed to crystal systems"""
    param_lat = ["a"]
    expr_lat = [
        "0",
        "a / 2.0",
        "a / 2.0",
        "a / 2.0",
        "0",
        "a / 2.0",
        "a / 2.0",
        "a / 2.0",
        "0",
    ]
    constr_lat = FixCartesianParametricRelations.from_expressions(
        indices=[0, 1, 2],
        params=param_lat,
        expressions=expr_lat,
        use_cell=True,
    )

    param_atom = []
    expr_atom = [
        "0.0",
        "0.0",
        "0.0",
        "0.25",
        "0.25",
        "0.25",
    ]
    constr_atom = FixScaledParametricRelations.from_expressions(
        indices=[0, 1],
        params=param_atom,
        expressions=expr_atom,
    )
    Si.set_constraint([constr_atom, constr_lat])
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Attempt to write a molecular system with geo_constrain=True
        Si.write(file, geo_constrain=True)
        assert len(w) == 1
        assert (
            str(w[-1].message)
            == "Setting scaled to True because a symmetry_block is detected."
        )

    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)
    assert len(Si.constraints) == len(new_atoms.constraints)
    assert str(Si.constraints[0]) == str(new_atoms.constraints[1])
    assert str(Si.constraints[1]) == str(new_atoms.constraints[0])


def test_wrap_Si(Si):
    """write fractional coords and check if structure was preserved"""
    Si.positions[0, 0] -= 0.015625
    Si.write(file, format=format, scaled=True, wrap=True)
    new_atoms = read(file)

    assert not np.allclose(Si.positions, new_atoms.positions)
    Si.wrap()
    assert np.allclose(Si.positions, new_atoms.positions)


def test_constraints_Si(Si):
    """Test that non-parmetric constraints are written and read in properly"""
    Si.set_constraint([FixAtoms(indices=[0]), FixCartesian(1, [1, 0, 1])])
    Si.write(file, format=format, scaled=True, wrap=False)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)
    assert len(Si.constraints) == len(new_atoms.constraints)
    assert str(Si.constraints[0]) == str(new_atoms.constraints[0])
    assert str(Si.constraints[1]) == str(new_atoms.constraints[1])


def test_cartesian_H2O(H2O):
    """write cartesian coords and check if structure was preserved for
    molecular systems"""
    H2O.write(file, format=format)
    new_atoms = read(file)
    assert np.allclose(H2O.positions, new_atoms.positions)


def test_scaled_H2O(H2O):
    """Attempt to write fractional coordinates and see if scaled is set to
    False and can be written properly"""
    with pytest.raises(
        ValueError,
        match="Requesting scaled for a calculation where scaled=True, "
            "but the system is not periodic",
    ):
        H2O.write(file, format=format, scaled=True, wrap=False)


def test_param_const_H2O(H2O):
    """Check to ensure if geo_constrain is True it does not affect the
    final geometry.in file for molecular systems"""
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Attempt to write a molecular system with geo_constrain=True
        H2O.write(file, geo_constrain=True)
        assert len(w) == 1
        assert (
            str(w[-1].message)
            == "Parameteric constraints can only be used in periodic systems."
        )

    new_atoms = read(file)
    assert np.allclose(H2O.positions, new_atoms.positions)


def test_velocities_H2O(H2O):
    """Confirm that the velocities are passed to the geometry.in file and
    can be read back in"""
    velocities = [(1.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]
    H2O.set_velocities(velocities)
    H2O.write(file, format=format, scaled=False, write_velocities=True)
    new_atoms = read(file)
    assert np.allclose(H2O.positions, new_atoms.positions)
    assert np.allclose(H2O.get_velocities(), new_atoms.get_velocities())


def test_info_str(H2O):
    """Confirm that the passed info_str is passed to the geometry.in file"""
    H2O.write(file, format=format, info_str="TEST INFO STR")
    with open(file) as fd:
        geometry_lines = fd.readlines()
        print(geometry_lines)
        assert "# Additional information:" in geometry_lines[6]
        assert "# TEST INFO STR" in geometry_lines[7]


sample_geometry_1 = """\
lattice_vector 4.5521460059804628 0.0000000000000000 0.0000000000000000
lattice_vector -2.2760730029902314 3.9422740829149499 0.000 # Dummy comment
lattice_vector 0.0000000000000000 0.0000000000000000 7.1603474299999998
atom_frac 0.0000000000000000 0.0000000000000000 0.000000 Pb # Dummy comment
atom_frac 0.6666666666666666 0.3333333333333333 0.7349025600000001 I
atom_frac 0.3333333333333333 0.6666666666666666 0.2650974399999999 I
#=======================================================
# Parametric constraints
#=======================================================
symmetry_n_params 3 2 1
symmetry_params a c d0_z
symmetry_lv a, 0, 0
symmetry_lv -0.5*a, 0.8660254037844*a, 0
symmetry_lv 0, 0, c
symmetry_frac 0, 0, 0
symmetry_frac 0.6666666666667, 0.3333333333333, 1.0-d0_z
symmetry_frac 0.3333333333333, 0.6666666666667, d0_z
"""

sample_geometry_2 = """\
atom 0.0000000000000000 0.0000000000000000 0.0000000000000 Pb # Dummy comment
    constrain_relaxation .true.
atom 0.6666666666666666 0.3333333333333333 0.7349025600000001 I
    initial_moment 1
    initial_charge -1
atom 0.3333333333333333 0.6666666666666666 0.2650974399999999 I
    constrain_relaxation y
    initial_charge -1
    initial_moment 1
"""

expected_symbols = ["Pb", "I", "I"]
expected_scaled_positions = np.array(
    [
        [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
        [0.6666666666666666, 0.3333333333333333, 0.7349025600000000],
        [0.3333333333333333, 0.6666666666666666, 0.2650974400000000],
    ]
)
expected_charges = np.array([0, -1, -1])
expected_moments = np.array([0, 1, 1])
expected_lattice_vectors = np.array(
    [
        [4.5521460059804628, 0.0000000000000000, 0.0000000000000000],
        [-2.2760730029902314, 3.9422740829149499, 0.0000000000000000],
        [0.0000000000000000, 0.0000000000000000, 7.1603474299999998],
    ]
)


def test_parse_geometry_lines():
    lines = sample_geometry_1.splitlines()
    atoms = parse_geometry_lines(lines, "sample_geometry_1.in")
    assert all(atoms.symbols == expected_symbols)
    assert atoms.get_scaled_positions() == approx(expected_scaled_positions)
    assert atoms.get_cell()[:] == approx(expected_lattice_vectors)
    assert all(atoms.pbc)

    lines = sample_geometry_2.splitlines()
    atoms = parse_geometry_lines(lines, "sample_geometry_2.in")
    assert all(atoms.symbols == expected_symbols)
    assert atoms.get_scaled_positions() == approx(expected_scaled_positions)
    assert atoms.get_initial_charges() == approx(expected_charges)
    assert atoms.get_initial_magnetic_moments() == approx(expected_moments)
    assert all(atoms.pbc == [0, 0, 0])
    assert len(atoms.constraints) == 2
