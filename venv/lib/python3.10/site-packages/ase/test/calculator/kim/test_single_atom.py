from pytest import mark

from ase import Atoms


@mark.calculator_lite
def test_single_atom(KIM):
    """
    Check that repositioning a single atom inside the cell finishes correctly
    and returns 0 energy, this in particular checks that the neighborlist
    maintenance for KIM models works for this case (see
    calculators/kim/neighbourlist.py).
    """
    model = "ex_model_Ar_P_Morse_07C"

    # twice the cutoff length
    box_len = 1.1

    atoms = Atoms("Ar", cell=[[box_len, 0, 0],
                              [0, box_len, 0],
                              [0, 0, box_len]],
                  calculator=KIM(model))

    for positions in [(0., 0., 0.), (0., 0., 1)]:
        atoms.set_positions([positions])

        assert atoms.get_potential_energy() == 0
