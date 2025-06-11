import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.fd import FiniteDifferenceCalculator


def test_fd():
    """Test `FiniteDifferenceCalculator`."""
    atoms = bulk('Cu', cubic=True)
    atoms.rattle(0.1)

    atoms.calc = EMT()
    energy_analytical = atoms.get_potential_energy()
    forces_analytical = atoms.get_forces()
    stress_analytical = atoms.get_stress()

    atoms.calc = FiniteDifferenceCalculator(EMT())
    energy_numerical = atoms.get_potential_energy()
    forces_numerical = atoms.get_forces()
    stress_numerical = atoms.get_stress()

    # check if `numerical` energy is exactly equal to `analytical`
    assert energy_numerical == energy_analytical

    # check if `numerical` forces are *not* exactly equal to `analytical`
    assert np.any(forces_numerical != forces_analytical)
    assert np.any(stress_numerical != stress_analytical)

    np.testing.assert_allclose(forces_numerical, forces_analytical)
    np.testing.assert_allclose(stress_numerical, stress_analytical)
