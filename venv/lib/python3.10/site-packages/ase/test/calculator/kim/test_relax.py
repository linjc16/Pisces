import numpy as np
from pytest import mark

from ase.cluster import Icosahedron
from ase.optimize import BFGS


@mark.calculator_lite
def test_relax(KIM, testdir):
    """
    Test that a static relaxation that requires multiple neighbor list
    rebuilds can be carried out successfully.  This is verified by relaxing
    an icosahedral cluster of atoms and checking that the relaxed energy
    matches a known precomputed value for an example model.
    """

    energy_ref = -0.56  # eV

    # Create structure and calculator
    atoms = Icosahedron("Ar", latticeconstant=3.0, noshells=2)
    calc = KIM("ex_model_Ar_P_Morse_07C")
    atoms.calc = calc

    with BFGS(atoms, maxstep=0.04, alpha=70.0, logfile=None) as opt:
        opt.run(fmax=0.01)  # eV/angstrom

    assert np.isclose(atoms.get_potential_energy(), energy_ref, atol=0.05)
