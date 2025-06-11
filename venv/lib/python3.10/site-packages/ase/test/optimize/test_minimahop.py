import pytest

from ase import Atom, Atoms
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.optimize.minimahopping import MinimaHopping


@pytest.mark.optimize()
def test_minimahop(asap3, testdir):
    # Make Pt 111 slab with Cu2 adsorbate.
    atoms = fcc111('Pt', (2, 2, 1), vacuum=7., orthogonal=True)
    adsorbate = Atoms([Atom('Cu', atoms[2].position + (0., 0., 2.5)),
                       Atom('Cu', atoms[2].position + (0., 0., 5.0))])
    atoms.extend(adsorbate)

    # Constrain the surface to be fixed and a Hookean constraint between
    # the adsorbate atoms.
    #
    # Actually: Why?  The test passes whether we set them or not, which
    # means we don't gain anything by including them.
    # I have disabled the hookeans to reduce test runtime.  --askhl
    #
    constraints = [FixAtoms(indices=[atom.index for atom in atoms if
                                     atom.symbol == 'Pt'])]
    # constraints = [FixAtoms(indices=[atom.index for atom in atoms if
    #                                  atom.symbol == 'Pt']),
    #                Hookean(a1=4, a2=5, rt=2.6, k=15.),
    #                Hookean(a1=4, a2=(0., 0., 1., -15.), k=15.)]
    atoms.set_constraint(constraints)

    # Set the calculator.
    calc = asap3.EMT()
    atoms.calc = calc

    # Instantiate and run the minima hopping algorithm.
    hop = MinimaHopping(atoms,
                        Ediff0=2.5,
                        T0=2000.,
                        beta1=1.2,
                        beta2=1.2,
                        mdmin=1)
    hop(totalsteps=3)
    # Test ability to restart and temperature stopping.
    hop(maxtemp=3000)
