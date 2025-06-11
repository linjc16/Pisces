import numpy as np
import pytest

from ase import Atoms


@pytest.mark.calculator_lite()
@pytest.mark.calculator("lammpslib")
def test_lammpslib_small_nonperiodic(factory, dimer_params, calc_params_NiH):
    """Test that lammpslib handle nonperiodic cases where the cell size
    in some directions is small (for example for a dimer)"""
    # Make a dimer in a nonperiodic box
    dimer = Atoms(**dimer_params)

    # Set of calculator
    calc = factory.calc(**calc_params_NiH)
    dimer.calc = calc

    # Check energy
    energy_ref = -1.10756669119
    energy = dimer.get_potential_energy()
    print(f"Computed energy: {energy}")
    assert energy == pytest.approx(energy_ref, rel=1e-4)

    # Check forces
    forces_ref = np.array(
        [[-0.9420162329811532, 0.0, 0.0], [+0.9420162329811532, 0.0, 0.0]]
    )
    forces = dimer.get_forces()
    print("Computed forces:")
    print(np.array2string(forces))
    assert forces == pytest.approx(forces_ref, rel=1e-4)

    energies = dimer.get_potential_energies()
    assert len(energies) == len(dimer)
    assert sum(energies) == pytest.approx(energy_ref, rel=1e-4)
    assert energies[0] == pytest.approx(energies[1], rel=1e-4)
