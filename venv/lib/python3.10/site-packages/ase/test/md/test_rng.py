"""Tests for legacy and modern NumPy PRNGs."""
import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md import Andersen
from ase.units import fs


@pytest.fixture(name='atoms')
def _fixture_atoms():
    atoms = bulk('Cu') * (2, 2, 2)
    atoms.calc = EMT()
    return atoms


@pytest.mark.parametrize(
    'rng',
    [np.random.RandomState(0), np.random.default_rng(42)],
)
def test_andersen(atoms, rng):
    """Test legacy and modern NumPy PRNGs for `Andersen`."""
    with Andersen(
        atoms,
        timestep=1.0 / fs,
        temperature_K=1000.0,
        andersen_prob=0.01,
        rng=rng,
    ) as md:
        md.run(5)
