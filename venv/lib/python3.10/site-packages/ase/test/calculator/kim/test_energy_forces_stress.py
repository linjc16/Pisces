import numpy as np
from pytest import mark

from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)
from ase.lattice.cubic import FaceCenteredCubic


@mark.calculator_lite
def test_energy_forces_stress(KIM, testdir):
    """
    To test that the calculator can produce correct energy and forces.  This
    is done by comparing the energy for an FCC argon lattice with an example
    model to the known value; the forces/stress returned by the model are
    compared to numerical estimates via finite difference.
    """

    # Create an FCC atoms crystal
    atoms = FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=(1, 1, 1),
        symbol="Ar",
        pbc=(1, 0, 0),
        latticeconstant=3.0,
    )

    # Perturb the x coordinate of the first atom by less than the cutoff
    # distance
    atoms.positions[0, 0] += 0.01

    calc = KIM("ex_model_Ar_P_Morse_07C")
    atoms.calc = calc

    # Get energy and analytical forces/stress from KIM model
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    # Previously computed energy for this configuration for this model
    energy_ref = 19.7196709065  # eV

    # Compute forces and virial stress numerically
    forces_numer = calculate_numerical_forces(atoms, eps=0.0001)
    stress_numer = calculate_numerical_stress(atoms, eps=0.0001, voigt=True)

    tol = 1e-6
    assert np.isclose(energy, energy_ref, tol)
    assert np.allclose(forces, forces_numer, tol)
    assert np.allclose(stress, stress_numer, tol)

    # This has been known to segfault
    atoms.set_pbc(True)
    atoms.get_potential_energy()
