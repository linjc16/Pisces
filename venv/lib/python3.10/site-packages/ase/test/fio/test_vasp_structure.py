# type: ignore
import io
import os
import unittest

import numpy as np
import pytest

import ase
import ase.build
import ase.io
from ase.build import graphene_nanoribbon
from ase.calculators.calculator import compare_atoms
from ase.constraints import (
    FixAtoms,
    FixedLine,
    FixedPlane,
    FixScaled,
    constrained_indices,
)
from ase.io.vasp import read_vasp_xdatcar, write_vasp_xdatcar


class TestXdatcarRoundtrip(unittest.TestCase):
    def setUp(self):
        self.outfile = 'NaCl.XDATCAR'

        self.NaCl = ase.build.bulk('NaCl', 'rocksalt', a=5.64)

    def tearDown(self):
        if os.path.isfile(self.outfile):
            os.remove(self.outfile)

    def assert_atoms_almost_equal(self, atoms, other, tol=1e-15):
        """Compare two Atoms objects, raising AssertionError if different"""
        system_changes = compare_atoms(atoms, other, tol=tol)

        if len(system_changes) > 0:
            raise AssertionError(
                'Atoms objects differ by {}'.format(', '.join(system_changes))
            )

    def assert_trajectory_almost_equal(self, traj1, traj2):
        self.assertEqual(len(traj1), len(traj2))
        for image, other in zip(traj1, traj2):
            self.assert_atoms_almost_equal(image, other)

    def test_roundtrip(self):
        # Create a series of translated cells
        trajectory = [self.NaCl.copy() for _ in range(5)]
        for i, atoms in enumerate(trajectory):
            atoms.set_scaled_positions(
                atoms.get_scaled_positions() + i * np.array([0.05, 0, 0.02])
            )
            atoms.wrap()

        ase.io.write(self.outfile, trajectory, format='vasp-xdatcar')
        roundtrip_trajectory = ase.io.read(self.outfile, index=':')
        self.assert_trajectory_almost_equal(trajectory, roundtrip_trajectory)

    def test_roundtrip_single_atoms(self):
        atoms = ase.build.bulk('Ge')
        ase.io.write(self.outfile, atoms, format='vasp-xdatcar')
        roundtrip_atoms = ase.io.read(self.outfile)
        self.assert_atoms_almost_equal(atoms, roundtrip_atoms)

    def test_typeerror(self):
        with self.assertRaises(TypeError):
            atoms = ase.build.bulk('Ge')
            write_vasp_xdatcar(self.outfile, atoms)
        with self.assertRaises(TypeError):
            not_atoms = 1
            ase.io.write(self.outfile, not_atoms, format='vasp-xdatcar')
        with self.assertRaises(TypeError):
            not_traj = [True, False, False]
            ase.io.write(self.outfile, not_traj, format='vasp-xdatcar')


def test_index():
    """Test if the `index` option works correctly"""
    atoms0 = ase.build.bulk('X', 'sc', a=1.0)
    atoms1 = atoms0.copy()
    atoms1.positions += 0.1
    images = (atoms0, atoms1)
    with io.StringIO() as buf:
        write_vasp_xdatcar(buf, images)

        buf.seek(0)
        atoms = read_vasp_xdatcar(buf, index=0)  # atoms0
        np.testing.assert_allclose(atoms.positions, [[0.0, 0.0, 0.0]])

        buf.seek(0)
        atoms = read_vasp_xdatcar(buf, index=-1)  # atoms1
        np.testing.assert_allclose(atoms.positions, [[0.1, 0.1, 0.1]])

        buf.seek(0)
        images = read_vasp_xdatcar(buf, index=None)  # atoms1
        np.testing.assert_allclose(atoms.positions, [[0.1, 0.1, 0.1]])

        buf.seek(0)
        images = read_vasp_xdatcar(buf, index=':')  # (atoms0, atoms1)
        assert isinstance(images, list)


# Start of tests for constraints
# VASP supports FixAtoms and FixScaled as well as FixedLine and
# FixedPlane if the direction is along a lattice vector. Test that
# these constraints are preserved when writing and reading POSCAR
# files.
indices_to_constrain = [0, 2]


@pytest.fixture()
def graphene_atoms():
    atoms = graphene_nanoribbon(2, 2, type="armchair", saturated=False)
    atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    return atoms


def poscar_roundtrip(atoms):
    """Write a POSCAR file, read it back and return the new atoms object"""
    atoms.write("POSCAR", direct=True)
    return ase.io.read("POSCAR")


def test_FixAtoms(graphene_atoms):
    atoms = graphene_atoms
    atoms.set_constraint(FixAtoms(indices=indices_to_constrain))
    new_atoms = poscar_roundtrip(atoms)

    # Assert that constraints are preserved
    assert isinstance(new_atoms.constraints[0], FixAtoms)
    assert np.all(new_atoms.constraints[0].index == indices_to_constrain)


def test_FixScaled(graphene_atoms):
    atoms = graphene_atoms
    atoms.set_constraint(FixScaled(indices_to_constrain, mask=[0, 1, 1]))
    new_atoms = poscar_roundtrip(atoms)

    # Assert that constraints are preserved
    assert np.all(constrained_indices(new_atoms) == indices_to_constrain)
    assert np.all(new_atoms.constraints[0].mask == [0, 1, 1])


@pytest.mark.parametrize('ConstraintClass', [FixedLine, FixedPlane])
def test_FixedLine_and_Plane(ConstraintClass, graphene_atoms):
    atoms = graphene_atoms
    atoms.set_constraint(
        ConstraintClass(indices=indices_to_constrain, direction=[1, 0, 0])
    )
    new_atoms = poscar_roundtrip(atoms)

    # FixedLine and FixedPlane are converted to FixScaled. During
    # a relaxation the results will be the same since FixScaled
    # are equivalent to the others if the direction in FixedLine
    # or FixedPlane is along a lattice vector.

    assert np.all(constrained_indices(new_atoms) == indices_to_constrain)


def test_write_read_velocities(graphene_atoms):
    vel = np.zeros_like(graphene_atoms.positions)
    vel = np.linspace(-1, 1, 3 * len(graphene_atoms)).reshape(-1, 3)
    graphene_atoms.set_velocities(vel)

    graphene_atoms.write('CONTCAR', direct=False)
    new_atoms = ase.io.read('CONTCAR')
    new_vel = new_atoms.get_velocities()

    assert np.allclose(vel, new_vel)
