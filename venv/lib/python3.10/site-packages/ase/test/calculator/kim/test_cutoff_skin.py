import numpy as np
import pytest

from ase import Atoms


@pytest.mark.calculator_lite()
def test_cutoff_skin(KIM, testdir):
    """
    To test that the calculator handles skin and cutoffs correctly.
    Specifically, note that the neighbor skin distance must be added onto
    both the model influence distance *and* each of the model cutoffs.  If
    the skin is not added onto the cutoffs, then an atom lying in between
    the largest cutoff and the skinned influence distance will not register
    as a neighbor if it hasn't already.

    The cutoff (and influence distance) for the model
    ex_model_Ar_P_Morse_07C is 8.15 Angstroms and the default skin distance
    when using the kimpy neighbor list library (which is the default when
    using a KIM portable model with this calculator) is 0.2 times the cutoff
    distance (1.63 Angstroms for this model).  Here, we construct a dimer
    with a separation falling just beyond the model cutoff but inside of the
    skinned influence distance.  We then compute the energy, which we expect
    to be zero in any case.  Next, we reduce the dimer separation by less
    than the skin distance so that the atoms fall within the cutoff of one
    another but without triggering a neighbor list rebuild.  If the atom had
    properly registered as a neighbor when it was outside of the cutoff but
    still inside of the skinned influence distance, then the energy in this
    case should be significantly far off from zero.  However, if the atom
    had failed to ever register as a neighbor, then we'll get zero once
    again.
    """
    calc = KIM("ex_model_Ar_P_Morse_07C")

    # Create dimer with separation just beyond cutoff distance.  We *want*
    # these atoms to register as neighbors of one another since they fall
    # within the skinned influence distance of 9.78 Angstroms.
    model_cutoff = 8.15
    skin_distance = 0.2 * model_cutoff
    distance_orig = model_cutoff + 0.1 * skin_distance
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [distance_orig, 0, 0]])
    atoms.calc = calc

    # Get energy -- should be zero
    e_outside_cutoff = atoms.get_potential_energy()

    # Now reduce the separation distance to something well within the model
    # cutoff -- should get something significantly non-zero
    atoms.positions[1, 0] -= 0.5 * skin_distance

    # Get new energy
    e_inside_cutoff = atoms.get_potential_energy()

    assert not np.isclose(e_outside_cutoff, e_inside_cutoff)
