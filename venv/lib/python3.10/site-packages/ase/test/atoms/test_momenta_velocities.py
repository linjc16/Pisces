import numpy as np
import pytest

from ase.build import molecule
from ase.constraints import FixAtoms, Hookean


@pytest.fixture()
def atoms():
    return molecule('CH3CH2OH')


def test_momenta_fixatoms(atoms):
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_momenta(np.ones(atoms.get_momenta().shape))
    desired = np.ones(atoms.get_momenta().shape)
    desired[0] = 0.
    actual = atoms.get_momenta()
    assert (actual == desired).all()


def test_momenta_hookean(atoms):
    atoms.set_constraint(Hookean(0, 1, rt=1., k=10.))
    atoms.set_momenta(np.zeros(atoms.get_momenta().shape))
    actual = atoms.get_momenta()
    desired = np.zeros(atoms.get_momenta().shape)
    # Why zero memoenta?  Should we not test something juicier?
    assert (actual == desired).all()


def test_get_set_velocities(atoms):
    shape = (len(atoms), 3)
    assert np.array_equal(atoms.get_velocities(), np.zeros(shape))

    rng = np.random.RandomState(17)
    v0 = rng.random(shape)
    atoms.set_velocities(v0)
    assert atoms.get_velocities() == pytest.approx(v0)
