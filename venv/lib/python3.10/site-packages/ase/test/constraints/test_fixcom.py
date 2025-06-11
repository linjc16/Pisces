"""Tests for FixCom."""
import numpy as np
import pytest

from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS


@pytest.fixture(name="atoms")
def fixture_atoms() -> Atoms:
    """fixture_atoms"""
    atoms = molecule('H2O')
    atoms.center(vacuum=4)
    atoms.calc = EMT()
    return atoms


@pytest.mark.optimize()
def test_center_of_mass_position(atoms: Atoms):
    """Test if the center of mass does not move."""
    cold = atoms.get_center_of_mass()
    atoms.set_constraint(FixCom())

    assert atoms.get_number_of_degrees_of_freedom() == 6

    with BFGS(atoms) as opt:
        opt.run(steps=5)

    cnew = atoms.get_center_of_mass()

    assert max(cnew - cold) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.optimize()
def test_center_of_mass_velocity(atoms: Atoms):
    """Test if the center-of-mass veloeicty is zero."""
    atoms.set_constraint(FixCom())

    # `adjust_momenta` of constaints are applied inside
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)

    velocity_com = atoms.get_momenta().sum(axis=0) / atoms.get_masses().sum()

    assert max(velocity_com) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.optimize()
def test_center_of_mass_force(atoms: Atoms):
    """Test if the corrected forces are along the COM-preserving direction."""
    rnd = np.random.default_rng(42)
    forces = rnd.random(size=atoms.positions.shape)
    FixCom().adjust_forces(atoms, forces)
    np.testing.assert_allclose(atoms.get_masses() @ forces, 0.0, atol=1e-12)
