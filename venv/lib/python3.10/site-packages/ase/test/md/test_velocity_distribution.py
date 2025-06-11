import numpy as np
import pytest

from ase.build import molecule
from ase.md.velocitydistribution import Stationary, ZeroRotation

norm = np.linalg.norm


@pytest.fixture()
def atoms():
    rng = np.random.RandomState(0)
    atoms = molecule('CH3CH2OH')
    momenta = -0.5 + rng.random((len(atoms), 3))
    atoms.set_momenta(momenta)
    return atoms


@pytest.fixture()
def stationary_atoms(atoms):
    atoms = atoms.copy()
    Stationary(atoms)
    return atoms


def propagate(atoms):
    dt = 0.1
    atoms = atoms.copy()
    velocities = atoms.get_velocities()
    displacement = velocities * dt
    atoms.positions += displacement
    return atoms


def test_stationary(atoms, stationary_atoms):
    assert norm(atoms.get_momenta().sum(axis=0)) > 0.1
    assert norm(stationary_atoms.get_momenta().sum(axis=0)) < 1e-13


def test_stationary_propagate(atoms, stationary_atoms):
    # Test that center of mass is stationary by time propagation
    prop_atoms = propagate(atoms)
    stationary_prop_atoms = propagate(stationary_atoms)
    com = atoms.get_center_of_mass()

    # Center of mass have moved (a bit, at least):
    assert norm(prop_atoms.get_center_of_mass() - com) > 1e-4
    assert norm(stationary_prop_atoms.get_center_of_mass() - com) < 1e-13


def test_zero_rotation(atoms):
    mom1 = atoms.get_angular_momentum()
    ZeroRotation(atoms)
    mom2 = atoms.get_angular_momentum()
    assert norm(mom1) > 0.1
    assert norm(mom2) < 1e-13


# Can we write a "propagation" test for zero rotation as well?
