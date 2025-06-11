import numpy as np
import pytest

from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixInternals
from ase.optimize.climbfixinternals import BFGSClimbFixInternals
from ase.vibrations import Vibrations


def setup_atoms():
    """Setup transition state search for the diffusion barrier for a Pt atom
    on a Pt surface."""
    atoms = fcc100('Pt', size=(2, 2, 1), vacuum=10.0)
    add_adsorbate(atoms, 'Pt', 1.611, 'hollow')
    atoms.set_constraint(FixAtoms(list(range(4))))  # freeze the slab
    return atoms


@pytest.mark.optimize()
@pytest.mark.parametrize('scaling', [0.0, 0.01])
def test_climb_fix_internals(scaling, testdir):
    """Climb along the constrained bondcombo coordinate while optimizing the
    remaining degrees of freedom after each climbing step.
    For the definition of constrained internal coordinates see the
    documentation of the classes FixInternals and ClimbFixInternals."""
    atoms = setup_atoms()
    atoms.calc = EMT()

    # Define reaction coordinate via linear combination of bond lengths
    reaction_coord = [[0, 4, 1.0], [1, 4, 1.0]]  # 1 * bond_1 + 1 * bond_2
    # Use current value `FixInternals.get_combo(atoms, reaction_coord)`
    # as initial value

    bondcombo = [None, reaction_coord]  # 'None' will converts to current value
    atoms.set_constraint([FixInternals(bondcombos=[bondcombo])]
                         + atoms.constraints)

    # Optimizer for transition state search along reaction coordinate
    opt = BFGSClimbFixInternals(atoms, climb_coordinate=reaction_coord,
                                optB_fmax_scaling=scaling)
    opt.run(fmax=0.05)  # Converge to a saddle point

    # Validate transition state by one imaginary vibrational mode
    vib = Vibrations(atoms, indices=[4])
    vib.run()
    assert ((np.imag(vib.get_energies()) > 0) == [True, False, False]).all()
# end example for documentation


@pytest.mark.optimize()
def test_initialization_with_different_constraints():
    """Remember to provide reaction coordinates as nested lists.
    Definitions in this example are arbitrary,
    the point is to check whether initialization is successful."""
    atoms = setup_atoms()

    # setup different constraints
    bond = FixInternals(bonds=[[None, [0, 1]]])
    angle = FixInternals(angles_deg=[[None, [0, 1, 2]]])
    dihedral = FixInternals(dihedrals_deg=[[None, [0, 1, 2, 3]]])
    bc = [[0, 4, -1.0], [1, 4, 1.0]]
    bondcombo = FixInternals(bondcombos=[[None, bc]])

    # test initialization of BFGSClimbFixInternals with different constraints
    coord = [[0, 1], [0, 1, 2], [0, 1, 2, 3], bc]
    for i, constr in enumerate([bond, angle, dihedral, bondcombo]):
        atoms.set_constraint(constr)
        BFGSClimbFixInternals(atoms, climb_coordinate=coord[i])
