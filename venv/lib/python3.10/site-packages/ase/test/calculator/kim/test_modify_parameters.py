import numpy as np
import pytest
from pytest import mark

from ase import Atoms


@mark.calculator_lite
def test_modify_parameters(KIM):
    """
    Check that the KIM calculator is capable of retrieving and updating model
    parameters correctly.  This is done by instantiating the calculator for a
    specific Lennard-Jones (LJ) potential, included with the KIM API, with a
    known cutoff for Mo-Mo interactions.  An Mo dimer is then constructed with
    a random separation that falls within the cutoff and its energy using the
    original potential parameters is computed.  Next, the original value of the
    "epsilon" parameter for Mo is retrieved.  The value of the parameter is
    then set to a scaling factor times the original value and the energy
    recomputed.  In the Lennard-Jones potential, the energy is directly
    proportional to the value of parameter "epsilon"; thus, the final energy
    computed is asserted to be approximately equal to the scaling factor times
    the original energy.
    """

    # In LennardJones612_UniversalShifted__MO_959249795837_003, the cutoff
    # for Mo interaction is 10.9759 Angstroms.
    cutoff = 10.9759

    # Create random dimer with separation < cutoff
    dimer_separation = np.random.RandomState(
        11).uniform(0.1 * cutoff, 0.6 * cutoff)
    atoms = Atoms("Mo" * 2, positions=[[0, 0, 0], [0, 0, dimer_separation]])

    calc = KIM("LennardJones612_UniversalShifted__MO_959249795837_003")
    atoms.calc = calc

    # Retrieve the original energy scaling parameter
    eps_orig = calc.get_parameters(epsilons=4879)["epsilons"][1]  # eV

    # Get the energy using the original parameter as a reference value
    E_orig = atoms.get_potential_energy()  # eV

    # Scale the energy scaling parameter and set this value to the calculator
    energy_scaling_factor = 2.0
    eps_modified = energy_scaling_factor * eps_orig
    calc.set_parameters(epsilons=[4879, eps_modified])

    # Get the energy after modifying the parameter
    E_modified = atoms.get_potential_energy()  # eV

    assert E_modified == pytest.approx(energy_scaling_factor * E_orig, rel=1e-4)
