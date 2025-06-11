import sys
from pathlib import Path
from subprocess import check_call, check_output

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.io.bundletrajectory import (
    BundleTrajectory,
    print_bundletrajectory_info,
)
from ase.io.pickletrajectory import PickleTrajectory

trajname = 'pickletraj.traj'


def test_raises():
    with pytest.raises(RuntimeError):
        PickleTrajectory(trajname, 'w')


@pytest.fixture()
def images():
    atoms = bulk('Ti') * (1, 2, 1)
    atoms.symbols = 'Au'
    atoms.calc = EMT()
    atoms1 = atoms.copy()
    atoms1.rattle()
    images = [atoms, atoms1]

    # Set all sorts of weird data:
    for i, atoms in enumerate(images):
        ints = np.arange(len(atoms)) + i
        floats = 1.0 + np.arange(len(atoms))
        atoms.set_tags(ints)
        atoms.set_initial_magnetic_moments(floats)
        atoms.set_initial_charges(floats)
        atoms.set_masses(floats)
        floats3d = 1.2 * np.arange(3 * len(atoms)).reshape(-1, 3)
        atoms.set_momenta(floats3d)
        atoms.info = {'hello': 'world'}
        atoms.calc = EMT()
        atoms.get_potential_energy()

    atoms.set_constraint(FixAtoms(indices=[0]))
    return [atoms, atoms1]


def read_images(filename):
    with PickleTrajectory(filename, _warn=False) as traj:
        return list(traj)


@pytest.fixture()
def trajfile(images):
    ptraj = PickleTrajectory(trajname, 'w', _warn=False)
    for image in images:
        ptraj.write(image)
    ptraj.close()
    return trajname


def assert_images_equal(images1, images2):
    assert len(images1) == len(images2), 'length mismatch'
    for atoms1, atoms2 in zip(images1, images2):
        differences = compare_atoms(atoms1, atoms2)
        assert not differences


def test_write_read_pickle(images, trajfile):
    images1 = read_images(trajfile)
    assert_images_equal(images, images1)


@pytest.mark.xfail(reason='bug: writes initial magmoms but reads magmoms '
                   'as part of calculator')
def test_write_read_bundle(images, bundletraj):
    images1 = read(bundletraj, ':')
    assert_images_equal(images, images1)


def test_append_pickle(images, trajfile):
    with PickleTrajectory(trajfile, 'a', _warn=False) as traj:
        for image in images:
            traj.write(image)

    images1 = read_images(trajfile)
    assert_images_equal(images * 2, images1)


@pytest.mark.xfail(reason='same as test_read_write_bundle')
def test_append_bundle(images, bundletraj):
    traj = BundleTrajectory(bundletraj, mode='a')
    assert len(read(bundletraj, ':')) == 2
    # write(bundletraj, images, append=True)
    for atoms in images:
        traj.write(atoms)
    traj.close()
    images1 = read(bundletraj, ':')
    assert len(images1) == 4
    # XXX Fix the magmoms/charges bug
    assert_images_equal(images * 2, images1)


def test_old_trajectory_conversion_utility(images, trajfile):
    trajpath = Path(trajfile)
    assert trajpath.exists()
    check_call([sys.executable, '-m', 'ase.io.trajectory', trajfile])
    oldtrajpath = trajpath.with_suffix('.traj.old')
    assert oldtrajpath.exists()
    assert trajpath.exists()  # New file should be where the old one was
    new_images = read(trajpath, ':', format='traj')
    assert_images_equal(images, new_images)


@pytest.fixture()
def bundletraj(images):
    fname = 'traj.bundle'
    write(fname, images, format='bundletrajectory')
    return fname


def test_bundletrajectory_info(images, bundletraj, capsys):
    print_bundletrajectory_info(bundletraj)
    output, _ = capsys.readouterr()

    natoms = len(images[0])
    expected_substring = f'Number of atoms: {natoms}'
    assert expected_substring in output

    # Same thing but via main():
    output2 = check_output([sys.executable,
                            '-m', 'ase.io.bundletrajectory', bundletraj],
                           encoding='ascii')
    assert expected_substring in output2
