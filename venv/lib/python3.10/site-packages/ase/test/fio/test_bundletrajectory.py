import os

import pytest

from ase import Atoms
from ase.io.bundletrajectory import (
    BundleTrajectory,
    read_bundletrajectory,
    write_bundletrajectory,
)


@pytest.fixture
def atoms():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


@pytest.fixture
def bundle_file(tmp_path):
    return os.path.join(tmp_path, "test.bundle")


def test_write_read(atoms, bundle_file):
    # Write atoms to BundleTrajectory
    traj = BundleTrajectory(bundle_file, "w", atoms=atoms)
    traj.write(atoms)
    traj.close()

    # Read atoms from BundleTrajectory
    traj = BundleTrajectory(bundle_file, "r")
    read_atoms = traj[0]
    traj.close()

    assert atoms == read_atoms


def test_append(atoms, bundle_file):
    # Write atoms to BundleTrajectory
    traj = BundleTrajectory(bundle_file, "w", atoms=atoms)
    traj.write(atoms)
    traj.close()

    # Append atoms to BundleTrajectory
    traj = BundleTrajectory(bundle_file, "a", atoms=atoms)
    traj.write(atoms)
    traj.close()

    # Read atoms from BundleTrajectory
    traj = BundleTrajectory(bundle_file, "r")
    assert len(traj) == 2
    read_atoms1 = traj[0]
    read_atoms2 = traj[1]
    traj.close()

    assert atoms == read_atoms1
    assert atoms == read_atoms2


def test_append_to_empty_bundle(atoms, bundle_file):
    # Create an empty BundleTrajectory
    traj = BundleTrajectory(bundle_file, "w")
    traj.close()

    # Append atoms to the empty BundleTrajectory
    traj = BundleTrajectory(bundle_file, "a", atoms=atoms)
    traj.write(atoms)
    traj.close()

    # Read atoms from BundleTrajectory
    traj = BundleTrajectory(bundle_file, "r")
    assert len(traj) == 1
    read_atoms = traj[0]
    traj.close()

    assert atoms == read_atoms


def test_append_to_nonexistent_bundle(atoms, bundle_file):
    # Append atoms to the nonexistent BundleTrajectory
    traj = BundleTrajectory(bundle_file, "a", atoms=atoms)
    traj.write(atoms)
    traj.close()

    # Read atoms from BundleTrajectory
    traj = BundleTrajectory(bundle_file, "r")
    assert len(traj) == 1
    read_atoms = traj[0]
    traj.close()

    assert atoms == read_atoms


def test_read_write_functions(atoms, bundle_file):
    # Write atoms using write_bundletrajectory
    write_bundletrajectory(bundle_file, atoms)

    # Read atoms using read_bundletrajectory
    read_atoms = next(read_bundletrajectory(bundle_file))

    assert atoms == read_atoms


def test_metadata(atoms, bundle_file):
    # Create a BundleTrajectory and write metadata
    traj = BundleTrajectory(bundle_file, "w")
    traj.write(atoms)
    traj.close()

    # Read metadata
    traj = BundleTrajectory(bundle_file, "r")
    metadata = traj.metadata
    traj.close()

    assert "format" in metadata
    assert metadata["format"] == "BundleTrajectory"
