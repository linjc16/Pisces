import pytest

from ase.constraints import FixAtoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.units import kB


def test_maxwellboltzmann():

    atoms = FaceCenteredCubic(size=(50, 50, 50), symbol="Cu", pbc=False)
    print("Number of atoms:", len(atoms))
    MaxwellBoltzmannDistribution(atoms, temperature_K=0.1 / kB)
    temp = atoms.get_kinetic_energy() / (1.5 * len(atoms))

    print("Temperature", temp, " (should be 0.1)")
    assert abs(temp - 0.1) < 1e-3


def test_maxwellboltzmann_dof():

    atoms = FaceCenteredCubic(size=(50, 50, 50), symbol="Cu", pbc=False)
    atoms.set_constraint(FixAtoms(range(250000)))

    MaxwellBoltzmannDistribution(atoms, temperature_K=1000)
    assert pytest.approx(atoms.get_temperature(), 5) == 1000

    MaxwellBoltzmannDistribution(atoms, temperature_K=1000, force_temp=True)
    assert pytest.approx(atoms.get_temperature(), 1e-8) == 1000

    Stationary(atoms, preserve_temperature=True)
    assert pytest.approx(atoms.get_temperature(), 1e-8) == 1000

    ZeroRotation(atoms, preserve_temperature=True)
    assert pytest.approx(atoms.get_temperature(), 1e-8) == 1000
