import numpy as np
import pytest

from ase.build import fcc111, molecule
from ase.build.attach import (
    attach,
    attach_randomly,
    attach_randomly_and_broadcast,
)
from ase.parallel import world


def test_attach_molecules():
    """Attach two molecules and check that their minimal distance
    is as required"""
    m1 = molecule('C6H6')
    m2 = molecule('NH3')

    distance = 2.
    m12 = attach(m1, m2, distance)
    dmin = np.linalg.norm(m12[15].position - m12[8].position)
    assert dmin == pytest.approx(distance, 1e-8)


def test_pbc():
    """Attach two molecules and check attachment considers pbc"""
    m1 = molecule('C6H6')
    m1.cell = (20, 1, 1)
    m1.translate((16, 0, 0))
    m1.pbc = (1, 0, 0)
    m2 = molecule('NH3')

    distance = 2.
    m12 = attach(m1, m2, distance)
    for atom in m12[-4:]:
        assert atom.position[0] < 2


def test_attach_to_surface():
    """Attach a molecule to a surafce at a given distance"""
    slab = fcc111('Al', size=(3, 2, 2), vacuum=10.0)
    mol = molecule('CH4')

    distance = 3.
    struct = attach(slab, mol, distance, (0, 0, 1))
    dmin = np.linalg.norm(struct[6].position - struct[15].position)
    assert dmin == pytest.approx(distance, 1e-8)


def test_attach_randomly():
    """Attach two molecules in random orientation."""
    m1 = molecule('C6H6')
    m2 = molecule('CF4')
    distance = 2.5

    if world.size > 1:
        "Check that the coordinates are correctly distributed from master."
        rng = np.random.RandomState(world.rank)  # ensure different seed
        atoms = attach_randomly_and_broadcast(m1, m2, distance, rng)

        p0 = 1. * atoms[-1].position
        world.broadcast(p0, 0)
        for i in range(1, world.size):
            pi = 1. * atoms[-1].position
            world.broadcast(pi, i)
            assert pi == pytest.approx(p0, 1e-8)

        "Check that every core has its own structure"
        rng = np.random.RandomState(world.rank)  # ensure different seed
        atoms = attach_randomly(m1, m2, distance, rng)
        p0 = 1. * atoms[-1].position
        world.broadcast(p0, 0)
        for i in range(1, world.size):
            pi = 1. * atoms[-1].position
            world.broadcast(pi, i)
            assert pi != pytest.approx(p0, 1e-8)

    rng = np.random.RandomState(42)  # ensure the same seed
    pos2_ac = np.zeros((5, 3))
    N = 25
    for _ in range(N):
        atoms = attach_randomly(m1, m2, distance, rng=rng)
        pos2_ac += atoms.get_positions()[12:, :]
    # the average position should be "zero" approximately
    assert (np.abs(pos2_ac / N) <= 1).all()
