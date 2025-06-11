import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.idealgas import IdealGas
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import kB


def test_idealgas():
    rng = np.random.RandomState(17)
    atoms = bulk('Kr').repeat((10, 10, 10))
    assert len(atoms) == 1000

    atoms.center(vacuum=100)
    atoms.calc = IdealGas()
    natoms = len(atoms)

    md_temp = 1000

    MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp, rng=rng)
    print(f"Temperature: {atoms.get_temperature()} K")

    with VelocityVerlet(atoms, timestep=0.1) as md:
        for _ in range(5):
            md.run(5)
            stress = atoms.get_stress(include_ideal_gas=True)
            stresses = atoms.get_stresses(include_ideal_gas=True)
            assert stresses.mean(0) == pytest.approx(stress)
            pressure = -stress[:3].sum() / 3
            pV = pressure * atoms.cell.volume
            NkT = natoms * kB * atoms.get_temperature()
            print(f"pV = {pV}  NkT = {NkT}")
            assert pV == pytest.approx(NkT, abs=1e-6)
