import numpy as np

from ase import Atoms
from ase.calculators.calculator import compare_atoms


def test_compare_atoms():
    """
    Check that Atoms.compare_atoms correctly accounts for the different
    types of system changes
    """

    # A system property that's an attribute of Atoms, but isn't in
    # Atoms.arrays (currently this is just 'cell' and 'pbc')
    cell1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cell2 = cell1 * 2
    atoms1 = Atoms(cell=cell1)
    atoms2 = Atoms(cell=cell2)
    assert set(compare_atoms(atoms1, atoms2)) == {"cell"}

    # A system property other than 'initial_charges' or 'initial_magmoms'
    # that exists in the `arrays` attribute of one Atoms object but not the
    # other
    atoms1 = Atoms()
    atoms2 = Atoms(numbers=[0], positions=[[0, 0, 0]])
    assert set(compare_atoms(atoms1, atoms2)) == {"positions", "numbers"}

    # A change in a system property that exists in the `arrays` attribute
    # of both Atoms objects passed into this function
    atoms1 = Atoms(numbers=[0], positions=[[0, 0, 0]])
    atoms2 = Atoms(numbers=[0], positions=[[1, 0, 0]])
    assert set(compare_atoms(atoms1, atoms2)) == {"positions"}

    # An excluded property (re-use atoms1 and atoms2 from previous check)
    assert set(compare_atoms(atoms1, atoms2,
               excluded_properties={"positions"})) == set()

    # Optional array (currently 'initial_charges' or 'initial_magmoms')
    # NOTE: Suppose you initialize an array of *zero charges* for atoms2
    #       but not atoms1.  The current compare_atoms function will still
    #       indicate that 'initial_charges' is in system_changes simply
    #       because it isn't in both of them.  This is despite the fact that
    #       if one were to call atoms1.get_initial_charges, you would get
    #       back an array of zeros.  However, this scenario should only ever
    #       occur rarely.
    atoms1 = Atoms(numbers=[0], positions=[[0, 0, 0]])
    atoms2 = Atoms(numbers=[0], positions=[[0, 0, 0]], charges=[1.13])
    assert set(compare_atoms(atoms1, atoms2)) == {"initial_charges"}
