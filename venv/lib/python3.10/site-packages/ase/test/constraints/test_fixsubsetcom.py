"""Tests for FixCom."""
import numpy as np
import pytest

from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixSubsetCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS


@pytest.fixture(name="atoms")
def fixture_atoms() -> Atoms:
    """fixture_atoms"""
    atoms0 = molecule('H2O')
    atoms0.positions -= 2.0
    atoms1 = molecule('H2O')
    atoms1.positions += 2.0
    atoms = atoms0 + atoms1
    atoms.calc = EMT()
    return atoms


@pytest.mark.optimize()
def test_center_of_mass_position(atoms: Atoms):
    """Test if the center of mass does not move."""
    indices = [3, 4, 5]
    cold = atoms.get_center_of_mass(indices=indices)
    atoms.set_constraint(FixSubsetCom(indices=indices))

    with BFGS(atoms) as opt:
        opt.run(steps=5)

    cnew = atoms.get_center_of_mass(indices=indices)

    assert max(cnew - cold) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.optimize()
def test_center_of_mass_velocity(atoms: Atoms):
    """Test if the center-of-mass veloeicty is zero."""
    indices = [3, 4, 5]
    atoms.set_constraint(FixSubsetCom(indices=indices))

    # `adjust_momenta` of constaints are applied inside
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)

    momenta = atoms.get_momenta()
    masses = atoms.get_masses()
    velocity_com = momenta[indices].sum(axis=0) / masses[indices].sum()

    assert max(velocity_com) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.optimize()
def test_center_of_mass_force(atoms: Atoms):
    """Test if the corrected forces are along the COM-preserving direction."""
    indices = [3, 4, 5]
    rnd = np.random.default_rng(42)
    forces = rnd.random(size=atoms.positions.shape)
    FixSubsetCom(indices=indices).adjust_forces(atoms, forces)
    tmp = atoms.get_masses()[indices] @ forces[indices]
    np.testing.assert_allclose(tmp, 0.0, atol=1e-12)
