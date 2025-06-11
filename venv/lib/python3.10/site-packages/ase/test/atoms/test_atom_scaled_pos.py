import numpy as np
import pytest

from ase.build import bulk


def get_spos(atoms):
    return atoms.get_scaled_positions(wrap=False)


@pytest.fixture()
def atoms():
    rng = np.random.RandomState(0)
    atoms = bulk('Ti') * (2, 2, 1)
    atoms.cell *= 0.9 + 0.2 * rng.random((3, 3))
    atoms.rattle(stdev=0.05, rng=rng)
    return atoms


@pytest.fixture()
def displacement(atoms):
    rng = np.random.RandomState(12345)
    return 0.1 * (rng.random((len(atoms), 3)) - 0.5)


@pytest.fixture()
def reference(displacement, atoms):
    return displacement + get_spos(atoms)


def test_abc_and_scaled_position(atoms):
    scaled = get_spos(atoms)
    for i, atom in enumerate(atoms):
        assert np.allclose(scaled[i], atom.scaled_position)
        assert np.allclose(scaled[i], [atom.a, atom.b, atom.c])


def test_set_scaled_position(atoms, displacement, reference):
    for i, atom in enumerate(atoms):
        atom.scaled_position += displacement[i]

    assert np.allclose(get_spos(atoms), reference)


def test_set_abc(atoms, displacement, reference):
    for i, atom in enumerate(atoms):
        atom.a += displacement[i, 0]
        atom.b += displacement[i, 1]
        atom.c += displacement[i, 2]

    assert np.allclose(get_spos(atoms), reference)
