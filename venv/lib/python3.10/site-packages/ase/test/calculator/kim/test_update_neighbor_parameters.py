import numpy as np
import pytest
from pytest import mark

from ase import Atoms


@mark.calculator_lite
def test_update_neighbor_parameters(KIM):
    """
    Check that the neighbor lists are updated properly when model parameters
    are updated. This is done by instantiating the calculator for a specific
    Lennard-Jones (LJ) potential, included with the KIM API, for Mo-Mo
    interactions.  First, an equally spaced collinear trimer is constructed
    whose separation distances are just over half the cutoff distance
    associated with Mo so that only nearest-neighbor interactions occur (the
    end atoms do not interact with one another) and the energy is computed.
    Next, the model's cutoff parameter associated with Mo is increased to a
    distance greater than the distance between the end atoms, so that their
    interaction will produce a (fairly small) non-zero value.  The energy is
    computed once again and it is verified that this energy differs
    significantly from the initial energy computed, thus implying that the
    neighbor lists of the end atoms must have been updated so as to contain one
    another.
    """

    # Create LJ calculator and disable energy shifting.  Otherwise, the energy
    # would always change when the cutoff changes, even if the neighbor lists
    # remained the same.
    calc = KIM(
        "LennardJones612_UniversalShifted__MO_959249795837_003",
        options={"neigh_skin_ratio": 0.0},
    )
    calc.set_parameters(shift=[0, 0])

    # Set all "cutoffs" parameters in the calculator to zero, except for the
    # one used for Mo.  This is necessary because we want to control the
    # influence distance, which is the maximum of all of the cutoffs.
    Mo_cutoff_index = 4879
    Mo_cutoff = calc.get_parameters(cutoffs=Mo_cutoff_index)["cutoffs"][1]
    cutoffs_extent = calc.parameters_metadata()["cutoffs"]["extent"]
    calc.set_parameters(
        cutoffs=[
            list(
                range(cutoffs_extent)),
            [0.0] *
            cutoffs_extent])
    calc.set_parameters(cutoffs=[Mo_cutoff_index, Mo_cutoff])

    # Create trimer such that nearest neighbor interactions occur:  each of the
    # end atoms see the middle atom, and the middle atom sees both end atoms,
    # but the end atoms do not interact with one another
    nearest_neighbor_separation = np.random.RandomState(11).uniform(
        0.6 * Mo_cutoff, 0.65 * Mo_cutoff
    )
    pos = [
        [0, 0, 0],
        [0, 0, nearest_neighbor_separation],
        [0, 0, 2 * nearest_neighbor_separation],
    ]
    trimer = Atoms("Mo" * 3, positions=pos)
    trimer.calc = calc

    eng_orig = trimer.get_potential_energy()

    # Update the cutoff parameter so that the end atoms will interact with one
    # another
    long_cutoff = 1.1 * \
        np.linalg.norm(np.array(pos[2][:]) - np.array(pos[0][:]))
    calc.set_parameters(cutoffs=[Mo_cutoff_index, long_cutoff])

    # Energy of the trimer after modifying cutoff
    eng_modified = trimer.get_potential_energy()

    # Check if the far atom is excluded when the original cutoff is used
    assert eng_modified != pytest.approx(eng_orig, rel=1e-4)
