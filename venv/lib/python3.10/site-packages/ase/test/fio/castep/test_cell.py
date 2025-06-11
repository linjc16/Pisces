"""Tests for CASTEP parsers"""
import io
import os
import re
import warnings

import numpy as np
import pytest

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import compare_atoms
from ase.constraints import FixAtoms, FixCartesian, FixedLine, FixedPlane
from ase.io import read, write
from ase.io.castep import read_castep_cell, write_castep_cell


# create mol with custom mass - from a list of positions or using
# ase.build.molecule
def write_read_atoms(atom, tmp_path):
    write(os.path.join(tmp_path, "castep_test.cell"), atom)
    return read(os.path.join(tmp_path, "castep_test.cell"))


# write to .cell and check that .cell has correct species_mass block in it
@pytest.mark.parametrize(
    "mol, custom_masses, expected_species, expected_mass_block",
    [
        ("CH4", {2: [1]}, ["C", "H:0", "H", "H", "H"], ["H:0 2.0"]),
        ("CH4", {2: [1, 2, 3, 4]}, ["C", "H", "H", "H", "H"], ["H 2.0"]),
        ("C2H5", {2: [2, 3]}, ["C", "C", "H:0",
         "H:0", "H", "H", "H"], ["H:0 2.0"]),
        (
            "C2H5",
            {2: [2], 3: [3]},
            ["C", "C", "H:0", "H:1", "H", "H", "H"],
            ["H:0 2.0", "H:1 3.0"],
        ),
    ],
)
def test_custom_mass_write(
    mol, custom_masses, expected_species, expected_mass_block, tmp_path
):

    custom_atoms = molecule(mol)
    atom_positions = custom_atoms.positions

    for mass, indices in custom_masses.items():
        for i in indices:
            custom_atoms[i].mass = mass

    atom_masses = custom_atoms.get_masses()
    # CASTEP IO can be noisy while handling keywords JSON
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        new_atoms = write_read_atoms(custom_atoms, tmp_path)

    # check atoms have been written and read correctly
    np.testing.assert_allclose(atom_positions, new_atoms.positions)
    np.testing.assert_allclose(atom_masses, new_atoms.get_masses())

    # check that file contains appropriate blocks
    with open(os.path.join(tmp_path, "castep_test.cell")) as f:
        data = f.read().replace("\n", "\\n")

    position_block = re.search(
        r"%BLOCK POSITIONS_ABS.*%ENDBLOCK POSITIONS_ABS", data)
    assert position_block

    pos = position_block.group().split("\\n")[1:-1]
    species = [p.split(" ")[0] for p in pos]
    assert species == expected_species

    mass_block = re.search(r"%BLOCK SPECIES_MASS.*%ENDBLOCK SPECIES_MASS", data)
    assert mass_block

    masses = mass_block.group().split("\\n")[1:-1]
    for line, expected_line in zip(masses, expected_mass_block):
        species_name, mass_read = line.split(' ')
        expected_species_name, expected_mass = expected_line.split(' ')
        assert pytest.approx(float(mass_read), abs=1e-6) == float(expected_mass)
        assert species_name == expected_species_name


# test setting a custom species on different atom before write
def test_custom_mass_overwrite(tmp_path):
    custom_atoms = molecule("CH4")
    custom_atoms[1].mass = 2

    # CASTEP IO is noisy while handling keywords JSON
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        atoms = write_read_atoms(custom_atoms, tmp_path)

    # test that changing masses when custom masses defined causes errors
    atoms[3].mass = 3
    with pytest.raises(ValueError,
                       match="Could not write custom mass block for H."):
        atoms.write(os.path.join(tmp_path, "castep_test2.cell"))


# suppress UserWarning due to keyword_tolerance
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestConstraints:
    """Test if the constraint can be recovered when writing and reading.

    Linear constraints in the CASTEP `.cell` format are flexible.
    The present `read_castep_cell` converts the linear constraints into single
    FixAtoms for the atoms for which all the three directions are fixed.
    Otherwise, it makes either `FixedLine` or `FixPlane` depending on the
    number of fixed directions for each atom.
    """

    # TODO: test also mask for FixCartesian

    @staticmethod
    def _make_atoms_ref():
        """water molecule"""
        atoms = molecule("H2O")
        atoms.cell = 10.0 * np.eye(3)
        atoms.pbc = True
        atoms.set_initial_magnetic_moments(len(atoms) * [0.0])
        return atoms

    def _apply_write_read(self, constraint) -> Atoms:
        atoms_ref = self._make_atoms_ref()
        atoms_ref.set_constraint(constraint)

        buf = io.StringIO()
        write_castep_cell(buf, atoms_ref)
        buf.seek(0)
        atoms = read_castep_cell(buf)

        assert not compare_atoms(atoms_ref, atoms)

        print(atoms_ref.constraints, atoms.constraints)

        return atoms

    def test_fix_atoms(self):
        """Test FixAtoms"""
        constraint = FixAtoms(indices=(1, 2))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixAtoms)
        assert all(atoms.constraints[0].index == constraint.index)

    def test_fix_cartesian_line(self):
        """Test FixCartesian along line"""
        # moved only along the z direction
        constraint = FixCartesian(0, mask=(1, 1, 0))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        for i, idx in enumerate(constraint.index):
            assert isinstance(atoms.constraints[i], FixedLine)
            assert atoms.constraints[i].index.tolist() == [idx]

    def test_fix_cartesian_plane(self):
        """Test FixCartesian in plane"""
        # moved only in the yz plane
        constraint = FixCartesian((1, 2), mask=(1, 0, 0))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 2
        for i, idx in enumerate(constraint.index):
            assert isinstance(atoms.constraints[i], FixedPlane)
            assert atoms.constraints[i].index.tolist() == [idx]

    def test_fix_cartesian_multiple(self):
        """Test multiple FixCartesian"""
        constraint = [FixCartesian(1), FixCartesian(2)]
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixAtoms)
        assert atoms.constraints[0].index.tolist() == [1, 2]

    def test_fixed_line(self):
        """Test FixedLine"""
        # moved only along the z direction
        constraint = FixedLine(0, direction=(0, 0, 1))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        for i, idx in enumerate(constraint.index):
            assert isinstance(atoms.constraints[i], FixedLine)
            assert atoms.constraints[i].index.tolist() == [idx]
            assert np.allclose(atoms.constraints[i].dir, constraint.dir)

    def test_fixed_plane(self):
        """Test FixedPlane"""
        # moved only in the yz plane
        constraint = FixedPlane((1, 2), direction=(1, 0, 0))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 2
        for i, idx in enumerate(constraint.index):
            assert isinstance(atoms.constraints[i], FixedPlane)
            assert atoms.constraints[i].index.tolist() == [idx]
            assert np.allclose(atoms.constraints[i].dir, constraint.dir)
